"""Constrained sampling for the Wyckoff representation.

The sampler enforces three things the representation makes enforceable:

  * the formula is exact. The benchmark formula is the primitive cell's, and the
    standard-setting cell holds centering(group) lattice points, so the orbits'
    multiplicities must sum to those counts times the centering factor;
  * a Wyckoff position with no free parameter is a single point set, so it can
    host at most one orbit;
  * no dead ends. Every space group, species and Wyckoff choice is admitted only
    if the counts left over can still be completed, which is checked exactly by
    a memoised search over the group's multiplicities.

Within those constraints the model chooses freely. Classifier-free guidance and
the BERTOS species restriction are applied exactly as in the per-atom sampler,
so the two decoders differ only in what the representation requires.
"""
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from pymatgen.core import Composition

from materium.generation_utils import cfg_blend, get_condition_dict, sample_next_token
from materium.wyckoff import centering, free_lattice_indices, group_table

# Node budget for one feasibility query. Ordinary formulas need a few hundred;
# the budget only bounds pathological low-symmetry cases, which are then assumed
# feasible -- the worst outcome is one rejected sample, not a hang.
_SEARCH_BUDGET = 20000


class _Budget(Exception):
    pass


def _feasible(number: int, remaining: Tuple[int, ...], used_fixed: frozenset) -> bool:
    counter = [0]

    @lru_cache(maxsize=None)
    def go(rem, used):
        counter[0] += 1
        if counter[0] > _SEARCH_BUDGET:
            raise _Budget()
        i = next((j for j, r in enumerate(rem) if r > 0), None)
        if i is None:
            return True
        r = rem[i]
        # Largest multiplicity first: a completion, when one exists, is usually
        # found on the first branch.
        for k, (mult, free) in sorted(enumerate(group_table(number)), key=lambda e: -e[1][0]):
            if mult > r or (not free and k in used):
                continue
            nxt = rem[:i] + (r - mult,) + rem[i + 1:]
            if go(nxt, used | {k} if not free else used):
                return True
        return False

    try:
        return go(tuple(remaining), frozenset(used_fixed))
    except _Budget:
        return True


@lru_cache(maxsize=4096)
def feasible_groups(counts: Tuple[int, ...]) -> Tuple[int, ...]:
    """Space groups in which a primitive cell with these element counts exists."""
    out = []
    for n in range(1, 231):
        c = centering(n)
        if _feasible(n, tuple(x * c for x in counts), frozenset()):
            out.append(n)
    return tuple(out)


def _bertos_species(tokenizer, formula, elements):
    """Per-element allowed species ids from BERTOS, with the per-atom sampler's fallback."""
    allowed = {}
    try:
        for spec in tokenizer.os_predictor.predict_formula(formula):
            if spec in tokenizer._species_to_id:
                allowed.setdefault(spec.split("|")[0], set()).add(tokenizer._species_to_id[spec])
    except Exception as e:
        print("Warning: BERTOS failed: %s; stoichiometry only." % e)
        return None
    for el in elements:
        if not allowed.get(el):
            allowed[el] = {sid for spec, sid in tokenizer._species_to_id.items()
                           if spec.split("|")[0] == el}
            print("BERTOS gave no valid token for %s; using all common OS as fallback." % el)
    return allowed


def generate_wyckoff_structure(
    model,
    tokenizer,
    max_len: int = 192,
    device="cpu",
    temperature: float = 1.0,
    top_p: float = 0.9,
    conditions: Optional[Dict] = None,
    classifier_guidance_weight: float = 0.0,
    csp_oxi_mode: Optional[str] = None,
    constrain_formula: bool = True,
    **_ignored,
):
    model.eval()
    sp = tokenizer._special_to_id
    sos, eos, atoms_tok, lattice_tok = sp["[SOS]"], sp["[EOS]"], sp["[ATOMS]"], sp["[LATTICE]"]
    V = tokenizer.vocab_size

    elements, counts = [], ()
    allowed_species = None
    formula = (conditions or {}).get("reduced_formula")
    if formula and constrain_formula:
        comp = Composition(formula)
        elements = [str(el) for el in comp.elements]
        counts = tuple(int(round(comp[el])) for el in comp.elements)
        if csp_oxi_mode == "bertos":
            allowed_species = _bertos_species(tokenizer, formula, elements)

    el_of_species = {sid: key.split("|")[0] for key, sid in tokenizer._species_to_id.items()}

    cond = get_condition_dict(model, conditions, device) if conditions else None

    def next_logits(seq):
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        if classifier_guidance_weight > 0.0 and cond:
            logits = cfg_blend(model(inp, conditions={}), model(inp, conditions=cond),
                               classifier_guidance_weight)
        else:
            logits = model(inp, conditions=cond)
        return logits[:, -1, :]

    def sample(seq, allowed_ids):
        if not allowed_ids:
            raise ValueError("no legal token at position %d" % len(seq))
        forbid = torch.ones(V, dtype=torch.bool, device=device)
        forbid[list(allowed_ids)] = False
        return sample_next_token(next_logits(seq), temperature=temperature, top_p=0.99,
                                 tail_bias=0.25, forbid_mask=forbid)

    bins = range(tokenizer.quant_offset, tokenizer.quant_offset + tokenizer.num_quant_bins)

    with torch.no_grad():
        seq = [sos]

        # 1. Space group. With a formula, only groups that can hold it.
        groups = feasible_groups(counts) if counts else range(1, 231)
        sg_tok = sample(seq, [tokenizer.sg_token(n) for n in groups])
        number = tokenizer.sg_of_token(sg_tok)
        seq += [sg_tok, atoms_tok]
        table = group_table(number)

        remaining = [x * centering(number) for x in counts]
        used_fixed = set()

        # 2. Orbits.
        while len(seq) < max_len:
            if counts and not any(remaining):
                break
            # Canonical sequences list orbits by electronegativity, so the next
            # orbit belongs to the least electronegative element still unplaced.
            # Elements of equal electronegativity are ordered by Wyckoff letter,
            # which is not known yet, so all of them stay legal.
            next_rank = None
            if counts and getattr(tokenizer, "canonical", False):
                next_rank = min(tokenizer.element_rank(e) for e, r in zip(elements, remaining) if r)
            legal = []
            for sid, el in el_of_species.items():
                if counts:
                    if el not in elements:
                        continue
                    i = elements.index(el)
                    if remaining[i] == 0:
                        continue
                    if next_rank is not None and tokenizer.element_rank(el) != next_rank:
                        continue
                    if allowed_species is not None and sid not in allowed_species.get(el, ()):
                        continue
                legal.append(sid)
            if not counts:
                legal.append(lattice_tok)
            sid = sample(seq, legal)
            if sid == lattice_tok:
                break
            seq.append(sid)
            el = el_of_species[sid]

            wp_legal = []
            for k, (mult, free) in enumerate(table):
                if not free and k in used_fixed:
                    continue
                if counts:
                    i = elements.index(el)
                    if mult > remaining[i]:
                        continue
                    nxt = list(remaining)
                    nxt[i] -= mult
                    if not _feasible(number, tuple(nxt), frozenset(used_fixed | ({k} if not free else set()))):
                        continue
                wp_legal.append(tokenizer.wp_token(number, k))
            wp_tok = sample(seq, wp_legal)
            seq.append(wp_tok)
            _, k = tokenizer._id_to_wp[wp_tok]
            mult, free = table[k]
            if not free:
                used_fixed.add(k)
            if counts:
                remaining[elements.index(el)] -= mult
            for _ in free:
                seq.append(sample(seq, bins))

        # 3. Lattice: exactly the crystal system's free parameters, then [EOS].
        seq.append(lattice_tok)
        for _ in free_lattice_indices(number):
            seq.append(sample(seq, bins))
        seq.append(eos)

    return seq
