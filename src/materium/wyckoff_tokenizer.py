"""Tokenizer for the Wyckoff representation.

Sequence:

    [SOS] [SG n] [ATOMS] ( [species] [WP n:k] [free bins...] )* [LATTICE] [free lattice bins] [EOS]

The space group comes first, so the model commits to a symmetry before placing
anything -- the prediction an auxiliary head was meant to force, made causally
from the formula alone. Each orbit then costs two tokens plus one bin per free
parameter, which on the training data is 6 numbers per structure against 57.

Built on CrystalTokenizer so that the species vocabulary, the oxidation-state
assignment (guess / TOSS) and the BERTOS predictor are the same objects the
existing runs used. The vocabulary keeps that tokenizer's layout as a prefix --
specials, species, then bins -- and appends the space-group and Wyckoff tokens
after it. That prefix matters: the formula condition embeds elements by looking
up the token-embedding table at their atomic numbers, so those rows must mean
what they meant before.
"""
import math
from collections import Counter
from typing import Dict, List, Optional

from pymatgen.core import Element, Species, Structure

from materium.tokenizer import CrystalTokenizer, SequenceOrder, SortingOrder
from materium.wyckoff import (
    SYMPREC,
    WyckoffCrystal,
    WyckoffSite,
    decode,
    encode,
    free_lattice_indices,
    full_lattice,
    group_table,
    to_primitive,
)
from materium.wyckoff_canonical import canonical_orbits, electronegativity

NUM_SPACE_GROUPS = 230


class WyckoffTokenizer(CrystalTokenizer):
    is_wyckoff = True

    # Cell lengths are binned on a log scale: standard-setting cells span 2.4 A
    # to over 100 A, and a linear grid fine enough at 3 A would waste most of its
    # bins on lengths that barely occur. 1024 bins over this range is ~0.4%.
    LOG_LENGTH_RANGE = (math.log(2.0), math.log(128.0))
    ANGLE_RANGE = (50.0, 150.0)

    def __init__(
        self,
        element_vocab: List[str],
        lattice_stats: Dict,
        num_quant_bins: int = 1024,
        include_oxidation: bool = True,
        oxidation_mode: str = "guess",
        allowed_oxidation_states: Optional[Dict[str, List[int]]] = None,
        fallback_to_zero_on_unseen: bool = True,
        symprec: float = SYMPREC,
        canonical: bool = False,
    ):
        super().__init__(
            element_vocab=element_vocab,
            lattice_stats=lattice_stats,
            num_quant_bins=num_quant_bins,
            sorting_order=SortingOrder.REVERSE_SPECIES,
            sequence_order=SequenceOrder.ATOMS_FIRST,
            include_oxidation=include_oxidation,
            oxidation_mode=oxidation_mode,
            allowed_oxidation_states=allowed_oxidation_states,
            fallback_to_zero_on_unseen=fallback_to_zero_on_unseen,
            legacy_bin_grid=False,
        )
        self.symprec = symprec
        # One description per crystal (wyckoff_canonical.py): orbits by
        # electronegativity then Wyckoff letter, normaliser setting and origin
        # fixed. Off, sequences are exactly those the wyck1 runs trained on.
        self.canonical = canonical

        self.sg_offset = self.vocab_size
        self.wp_offset = self.sg_offset + NUM_SPACE_GROUPS
        # Every Wyckoff position of every group gets its own token, so a token
        # means one position unambiguously and the decoder can mask to the chosen
        # group's positions without consulting context.
        self._wp_keys = [
            (n, k)
            for n in range(1, NUM_SPACE_GROUPS + 1)
            for k in range(len(group_table(n)))
        ]
        self._wp_to_id = {key: self.wp_offset + i for i, key in enumerate(self._wp_keys)}
        self._id_to_wp = {i: key for key, i in self._wp_to_id.items()}
        self.vocab_size = self.wp_offset + len(self._wp_keys)

        # Orbits in the most recent tokenize() whose sites were assigned more than
        # one oxidation state, resolved by majority. Read by diagnostics only.
        self.last_mixed_orbits = 0
        self.last_fallback = False
        self.last_canonical_fallback = False

    # ------------------------------------------------------------------ config
    def to_dict(self):
        return {
            "kind": "wyckoff",
            "element_vocab": self.element_vocab,
            "lattice_stats": self.lattice_stats,
            "num_quant_bins": self.num_quant_bins,
            "include_oxidation": self.include_oxidation,
            "oxidation_mode": self.oxidation_mode,
            "allowed_oxidation_states": self.allowed_oxidation_states,
            "fallback_to_zero_on_unseen": self.fallback_to_zero_on_unseen,
            "symprec": self.symprec,
            "canonical": self.canonical,
        }

    @classmethod
    def from_dict(cls, json_data):
        d = dict(json_data)
        d.pop("kind", None)
        return cls(**d)

    def token_layout(self):
        # The structured coordinate loss assumes fixed four-token atom blocks.
        # These blocks are variable-length, so no layout: training uses flat
        # cross-entropy, which is the objective these runs are meant to use.
        return None

    # ------------------------------------------------------------- token maps
    def sg_token(self, number: int) -> int:
        return self.sg_offset + number - 1

    def sg_of_token(self, token: int) -> Optional[int]:
        n = token - self.sg_offset + 1
        return n if 1 <= n <= NUM_SPACE_GROUPS else None

    def wp_token(self, number: int, index: int) -> int:
        return self._wp_to_id[(number, index)]

    def bin_token(self, b: int) -> int:
        return self.quant_offset + b

    def is_bin(self, token: int) -> bool:
        return 0 <= token - self.quant_offset < self.num_quant_bins

    # ------------------------------------------------------------ quantisation
    def _q(self, t: float) -> int:
        return self._quantize_unit(t)

    def _dq(self, b: int) -> float:
        return (b + 0.5) / self.num_quant_bins

    def _lattice_to_unit(self, idx: int, value: float) -> float:
        if idx < 3:
            lo, hi = self.LOG_LENGTH_RANGE
            return (math.log(max(value, 1e-6)) - lo) / (hi - lo)
        lo, hi = self.ANGLE_RANGE
        return (value - lo) / (hi - lo)

    def _unit_to_lattice(self, idx: int, t: float) -> float:
        if idx < 3:
            lo, hi = self.LOG_LENGTH_RANGE
            return math.exp(lo + t * (hi - lo))
        lo, hi = self.ANGLE_RANGE
        return lo + t * (hi - lo)

    def element_rank(self, element: str) -> float:
        """The orbit-ordering key of an element in canonical mode."""
        return electronegativity(element)

    def _lattice_bins(self, number: int, params):
        """Free lattice parameters as bins, or None if any falls off the grid --
        a clipped value would not decode to the same cell."""
        out = []
        for i in free_lattice_indices(number):
            t = self._lattice_to_unit(i, params[i])
            if not 0.0 <= t <= 1.0:
                return None
            out.append(self._q(t))
        return out

    def _species_id(self, site) -> int:
        key = self._get_site_species_token_key(site)
        if key not in self._species_to_id:
            key = "%s|+0" % site.specie.symbol
        return self._species_to_id[key]

    # --------------------------------------------------------------- tokenize
    @staticmethod
    def _as_p1(structure: Structure) -> WyckoffCrystal:
        """Lossless fallback: every atom its own orbit on P1's general position."""
        s = structure.get_reduced_structure(reduction_algo="niggli")
        s.remove_oxidation_states()
        return WyckoffCrystal(
            space_group=1,
            lattice=tuple(s.lattice.parameters),
            sites=[WyckoffSite(site.specie.symbol, 0, "1a", 1, [float(x) % 1.0 for x in site.frac_coords])
                   for site in s.sites],
        )

    def tokenize(self, structure: Structure) -> List[int]:
        # CrystalDataset maps every index through tokenize with no way to skip
        # one, so a structure pyxtal cannot analyse must still produce tokens.
        # P1 is exact for any structure -- it just carries no symmetry.
        try:
            wc = encode(structure, self.symprec)
            self.last_fallback = False
        except Exception:
            wc = self._as_p1(structure)
            self.last_fallback = True

        # Oxidation is assigned on the decoded standard cell, whose atoms are
        # labelled by orbit, so every orbit's sites are known exactly. All sites
        # of an orbit share one species token; if the assignment disagrees within
        # an orbit, the majority state is taken and the disagreement counted.
        std, orbit_of = decode(wc, return_orbits=True)
        std = self._assign_oxidation_states_if_needed(std)
        votes = [Counter() for _ in wc.sites]
        for site, k in zip(std.sites, orbit_of):
            votes[k][self._species_id(site)] += 1
        self.last_mixed_orbits = sum(1 for v in votes if len(v) > 1)
        n = wc.space_group
        sids = [votes[k].most_common(1)[0][0] for k in range(len(wc.sites))]

        self.last_canonical_fallback = False
        if self.canonical:
            res = canonical_orbits(
                n, std.lattice, std.frac_coords, orbit_of,
                [site.element for site in wc.sites], sids,
                lambda v: self._q(v % 1.0),
                lambda params: self._lattice_bins(n, params),
            )
            if res is not None:
                ordered, lattice = res
                tokens = [self._special_to_id["[SOS]"], self.sg_token(n), self._special_to_id["[ATOMS]"]]
                for k, wp_index, free in ordered:
                    tokens.append(sids[k])
                    tokens.append(self.wp_token(n, wp_index))
                    tokens.extend(self.bin_token(self._q(v % 1.0)) for v in free)
                tokens.append(self._special_to_id["[LATTICE]"])
                tokens.extend(self.bin_token(b) for b in self._lattice_bins(n, lattice))
                tokens.append(self._special_to_id["[EOS]"])
                return tokens
            # No candidate passed the exactness checks: write encode()'s own
            # description, which is still correct, just not canonical.
            self.last_canonical_fallback = True

        orbits = []
        for k, site in enumerate(wc.sites):
            sid = sids[k]
            free_bins = [self._q(v % 1.0) for v in site.free]
            key = (Element(site.element).Z, sid, site.wp_index, tuple(free_bins))
            orbits.append((key, sid, site.wp_index, free_bins))
        # Reverse species order, matching the sorting the existing runs used.
        orbits.sort(key=lambda o: o[0], reverse=True)

        tokens = [self._special_to_id["[SOS]"], self.sg_token(n), self._special_to_id["[ATOMS]"]]
        for _, sid, wp_index, free_bins in orbits:
            tokens.append(sid)
            tokens.append(self.wp_token(n, wp_index))
            tokens.extend(self.bin_token(b) for b in free_bins)

        tokens.append(self._special_to_id["[LATTICE]"])
        for i in free_lattice_indices(n):
            tokens.append(self.bin_token(self._q(self._lattice_to_unit(i, wc.lattice[i]))))
        tokens.append(self._special_to_id["[EOS]"])
        return tokens

    # ------------------------------------------------------------- detokenize
    def parse(self, tokens: List[int]):
        """Tokens -> (WyckoffCrystal, oxidation state per orbit)."""
        toks = [t for t in tokens if t not in (self._special_to_id["[PAD]"], self._special_to_id["[SOS]"])]
        if not toks:
            raise ValueError("empty sequence")
        number = self.sg_of_token(toks[0])
        if number is None:
            raise ValueError("sequence does not start with a space-group token")
        table = group_table(number)
        i = 1
        if i >= len(toks) or toks[i] != self._special_to_id["[ATOMS]"]:
            raise ValueError("missing [ATOMS] after the space group")
        i += 1

        sites, oxs = [], []
        while i < len(toks) and toks[i] != self._special_to_id["[LATTICE]"]:
            sid = toks[i]
            if sid not in self._id_to_species:
                raise ValueError("expected a species token, got %d" % sid)
            el, ox = self._id_to_species[sid].split("|") if self.include_oxidation \
                else (self._id_to_species[sid], "0")
            if i + 1 >= len(toks) or toks[i + 1] not in self._id_to_wp:
                raise ValueError("species not followed by a Wyckoff token")
            wn, wk = self._id_to_wp[toks[i + 1]]
            if wn != number:
                raise ValueError("Wyckoff token of group %d inside group %d" % (wn, number))
            mult, free_idx = table[wk]
            bins = toks[i + 2:i + 2 + len(free_idx)]
            if len(bins) != len(free_idx) or not all(self.is_bin(b) for b in bins):
                raise ValueError("Wyckoff position %d:%d needs %d free parameters" % (wn, wk, len(free_idx)))
            sites.append(WyckoffSite(el, wk, "", mult,
                                     [self._dq(b - self.quant_offset) for b in bins]))
            oxs.append(int(ox))
            i += 2 + len(free_idx)

        if i >= len(toks):
            raise ValueError("no [LATTICE] section")
        i += 1
        lat_idx = free_lattice_indices(number)
        bins = toks[i:i + len(lat_idx)]
        if len(bins) != len(lat_idx) or not all(self.is_bin(b) for b in bins):
            raise ValueError("lattice needs %d parameters" % len(lat_idx))
        free = [self._unit_to_lattice(j, self._dq(b - self.quant_offset)) for j, b in zip(lat_idx, bins)]
        if not sites:
            raise ValueError("no orbits")
        return WyckoffCrystal(number, full_lattice(number, free), sites), oxs

    def detokenize(self, tokens: List[int]) -> Structure:
        wc, oxs = self.parse(tokens)
        structure, orbit_of = decode(wc, return_orbits=True)
        if self.include_oxidation:
            structure = Structure(
                structure.lattice,
                [Species(s.specie.symbol, oxs[k]) for s, k in zip(structure.sites, orbit_of)],
                structure.frac_coords,
            )
        # The sampler places centering x formula atoms in the conventional cell;
        # what is written out is the primitive cell, which holds the formula.
        return to_primitive(structure, wc.space_group)
