#!/usr/bin/env python3
"""Pre-training gate for the Wyckoff tokenizer and its constrained sampler.

Three checks, all before any GPU time is spent:

1. Tokenizer round-trip, through the REAL tokenize/detokenize -- log-scale
   lattice bins, free lattice parameters only, oxidation per orbit -- scored with
   the benchmark's StructureMatcher. wyckoff_roundtrip.py validated the
   representation; this validates the tokens.

2. Count consistency on the 180 benchmark structures: the orbits' multiplicities
   must equal the formula's primitive counts times the group's centering. The
   sampler enforces exactly that relation, so if it fails here, constrained
   generation would force the wrong number of atoms.

3. Sampler smoke test with a randomly initialised model. Whatever an untrained
   model emits, the constraints must still deliver a sequence that detokenizes
   to exactly the requested formula. A failure here is a decoder bug, and no
   amount of training would hide it.

    python wyckoff_tokenizer_check.py <mp20 split dir> <groundtruth dir> <ox mode> [n]
"""
import collections
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Lattice, Structure

from materium.model import ConditionConfig, LLamaTransformer, ModelArgs
from materium.wyckoff import centering
from materium.wyckoff_generation import feasible_groups, generate_wyckoff_structure
from materium.wyckoff_tokenizer import WyckoffTokenizer

warnings.filterwarnings("ignore")
MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
ELEMENT_VOCAB = [Element.from_Z(i).symbol for i in range(1, 100)]
LATTICE_STATS = {"a": (2.0, 10.0), "b": (2.0, 12.5), "c": (2.0, 20.0),
                 "alpha": (60.0, 120.0), "beta": (60.0, 120.0), "gamma": (60.0, 120.0)}


def load_split(d):
    pos = np.load(os.path.join(d, "pos.npy"))
    cell = np.load(os.path.join(d, "cell.npy"))
    z = np.load(os.path.join(d, "atomic_numbers.npy"))
    n = np.load(os.path.join(d, "num_atoms.npy"))
    off = np.concatenate([[0], np.cumsum(n)])
    for i in range(len(n)):
        # pos.npy is fractional (see wyckoff_roundtrip.py).
        yield Structure(Lattice(cell[i]), z[off[i]:off[i + 1]].tolist(),
                        pos[off[i]:off[i + 1]], coords_are_cartesian=False)


def roundtrip(tok, name, structs):
    t0 = time.time()
    ok = fails = mixed = mixed_structs = 0
    lengths = []
    bad = []
    for s in structs:
        try:
            toks = tok.tokenize(s)
            lengths.append(len(toks))
            mixed += tok.last_mixed_orbits
            mixed_structs += tok.last_mixed_orbits > 0
            dec = tok.detokenize(toks)
            ref = s.copy()
            ref.remove_oxidation_states()
            d2 = dec.copy()
            d2.remove_oxidation_states()
            if MATCHER.fit(ref, d2):
                ok += 1
            else:
                bad.append(("match", s.composition.reduced_formula))
        except Exception as e:
            fails += 1
            bad.append(("error", s.composition.reduced_formula, repr(e)[:100]))
    n = len(structs)
    print("=" * 70)
    print("%s [%s]: %d structures, %.0f s" % (name, tok.oxidation_mode, n, time.time() - t0))
    print("  token round-trip match  %d (%.2f%%)   errors %d" % (ok, 100.0 * ok / n, fails))
    if lengths:
        print("  sequence length  mean %.1f  max %d  (model max_seq_len 192)"
              % (np.mean(lengths), max(lengths)))
    print("  structures with an orbit of mixed oxidation: %d (%.2f%%), orbits %d"
          % (mixed_structs, 100.0 * mixed_structs / n, mixed))
    for b in bad[:8]:
        print("    ", b)
    return ok, n


def counts_check(tok, gt_dir):
    bad = 0
    for f in sorted(os.listdir(gt_dir)):
        if not f.endswith(".cif"):
            continue
        s = Structure.from_file(os.path.join(gt_dir, f))
        wc, _ = tok.parse(tok.tokenize(s))
        formula = Composition(f[:-4])
        expect = {str(e): int(round(formula[e])) * centering(wc.space_group) for e in formula.elements}
        got = collections.Counter()
        for site in wc.sites:
            got[site.element] += site.multiplicity
        if dict(got) != expect:
            bad += 1
            if bad <= 8:
                print("    count mismatch %s sg %d: orbits %s vs formula x centering %s"
                      % (f[:-4], wc.space_group, dict(got), expect))
        elif wc.space_group not in feasible_groups(tuple(int(round(formula[e])) for e in formula.elements)):
            bad += 1
            print("    %s: true group %d excluded by feasible_groups" % (f[:-4], wc.space_group))
    print("=" * 70)
    print("benchmark count consistency: %d mismatches of 180" % bad)


def sampler_smoke(tok, gt_dir, n_formulas=40, per=3):
    torch.manual_seed(0)
    dim = 64
    cfg = {
        "space_group": ConditionConfig(proj_layer_type="embedding", input_dim=230, out_dim=dim),
        "reduced_formula": ConditionConfig(proj_layer_type="formula", input_dim=20, out_dim=dim),
    }
    params = ModelArgs(dim=dim, n_layers=2, n_heads=4, vocab_size=len(tok), multiple_of=32,
                       max_seq_len=192, pad_id=tok._special_to_id["[PAD]"], condition_config=cfg)
    model = LLamaTransformer(params).eval()

    formulas = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".cif"))
    random.seed(0)
    formulas = random.sample(formulas, n_formulas)
    ok = bad = 0
    t0 = time.time()
    for f in formulas:
        target = Composition(f)
        for _ in range(per):
            try:
                seq = generate_wyckoff_structure(model, tok, conditions={"reduced_formula": f},
                                                 classifier_guidance_weight=2.5,
                                                 csp_oxi_mode="bertos", constrain_formula=True)
                s = tok.detokenize(seq)
                wc, _ = tok.parse(seq)
                want = {str(e): int(round(target[e])) for e in target.elements}
                # Element counts: the decoded species carry oxidation states.
                ec = s.composition.element_composition
                got = {str(e): int(round(ec[e])) for e in ec.elements}
                if got == want:
                    ok += 1
                else:
                    bad += 1
                    print("    composition %s sg %d: got %s want %s" % (f, wc.space_group, got, want))
            except Exception as e:
                bad += 1
                print("    %s: %r" % (f, e))
    print("=" * 70)
    print("sampler smoke (random model, bertos): %d exact-formula structures, %d failures, %.1f s/sample"
          % (ok, bad, (time.time() - t0) / max(ok + bad, 1)))
    return ok, bad


def main():
    split, gt_dir, ox = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    tok = WyckoffTokenizer(element_vocab=ELEMENT_VOCAB, lattice_stats=LATTICE_STATS,
                           num_quant_bins=1024, oxidation_mode=ox)
    print("vocab %d (species from %d, bins from %d, groups from %d, Wyckoff from %d)"
          % (len(tok), len(tok.special_tokens), tok.quant_offset, tok.sg_offset, tok.wp_offset))

    random.seed(0)
    train = list(load_split(split))
    roundtrip(tok, "mp_20_heldout/train sample", random.sample(train, n))
    gt = [Structure.from_file(os.path.join(gt_dir, f))
          for f in sorted(os.listdir(gt_dir)) if f.endswith(".cif")]
    roundtrip(tok, "CSPBench 180", gt)
    counts_check(tok, gt_dir)
    if ox == "guess":
        sampler_smoke(tok, gt_dir)


def smoke_only():
    # argv: --smoke <split> <groundtruth> ...
    gt_dir = sys.argv[3]
    tok = WyckoffTokenizer(element_vocab=ELEMENT_VOCAB, lattice_stats=LATTICE_STATS,
                           num_quant_bins=1024, oxidation_mode="guess")
    sampler_smoke(tok, gt_dir, n_formulas=60, per=3)


if __name__ == "__main__":
    if sys.argv[1] == "--smoke":
        smoke_only()
    else:
        main()
