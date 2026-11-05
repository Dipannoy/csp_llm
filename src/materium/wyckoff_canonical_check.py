#!/usr/bin/env python3
"""Pre-training gate for the canonical Wyckoff tokenizer.

1. Token round-trip through the real tokenizer (canonical=True), as in
   wyckoff_tokenizer_check.py: the chosen description must decode to the crystal.

2. Invariance. Each structure is re-entered as an equivalent cell -- another
   basis, atoms reordered, the whole crystal shifted -- and must give the same
   tokens. Measured for the current and the canonical tokenizer side by side.
   Residual differences of a bin or so come from spglib refining a slightly
   imperfect structure differently, not from the representation.

3. Canonical fallbacks: structures for which no candidate passed the exactness
   checks and encode()'s own description was written instead.

4. Sampler smoke with a random model: every sequence must still detokenize to
   exactly the requested formula under the electronegativity-order constraint.

    python wyckoff_canonical_check.py <mp20 split dir> <groundtruth dir> <ox mode> [n] [n_invariance]
"""
import os
import random
import sys
import time
import warnings

import numpy as np
from pymatgen.core import Structure

from materium.wyckoff_tokenizer import WyckoffTokenizer
from materium.wyckoff_tokenizer_check import ELEMENT_VOCAB, LATTICE_STATS, load_split, roundtrip, sampler_smoke

warnings.filterwarnings("ignore")
BASES = [np.eye(3, dtype=int), np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
         np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])]


def perturb(s, rng):
    p = s.copy()
    p.make_supercell(BASES[rng.randrange(len(BASES))])
    order = list(range(len(p)))
    rng.shuffle(order)
    p = Structure(p.lattice, [p[i].specie for i in order], [p[i].frac_coords for i in order])
    p.translate_sites(list(range(len(p))), [rng.random() for _ in range(3)], frac_coords=True)
    return p


def invariance(toks, name, structs, k=3):
    rng = random.Random(0)
    results = {}
    perts = [[perturb(s, rng) for _ in range(k)] for s in structs]
    for label, tok in toks:
        t0 = time.time()
        same = total = changed = fallbacks = errors = 0
        for s, ps in zip(structs, perts):
            try:
                base = tok.tokenize(s)
                fallbacks += getattr(tok, "last_canonical_fallback", False)
                ok = True
                for p in ps:
                    total += 1
                    hit = tok.tokenize(p) == base
                    same += hit
                    ok &= hit
                changed += not ok
            except Exception:
                errors += 1
        results[label] = (same / max(total, 1), fallbacks / max(len(structs), 1), errors)
        print("%s [%s, %s]: identical tokens %d/%d (%.1f%%), structures with any change %d/%d, "
              "canonical fallbacks %d, errors %d, %.2f s/structure"
              % (name, label, tok.oxidation_mode, same, total, 100.0 * same / max(total, 1), changed,
                 len(structs), fallbacks, errors, (time.time() - t0) / len(structs)), flush=True)
    return results


def main():
    split, gt_dir, ox = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    n_inv = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    kw = dict(element_vocab=ELEMENT_VOCAB, lattice_stats=LATTICE_STATS, num_quant_bins=1024, oxidation_mode=ox)
    canon = WyckoffTokenizer(canonical=True, **kw)
    current = WyckoffTokenizer(canonical=False, **kw)
    assert WyckoffTokenizer.from_dict(canon.to_dict()).canonical
    assert not WyckoffTokenizer.from_dict({k: v for k, v in current.to_dict().items() if k != "canonical"}).canonical

    random.seed(0)
    train = list(load_split(split))
    sample = random.sample(train, n)
    gt = [Structure.from_file(os.path.join(gt_dir, f)) for f in sorted(os.listdir(gt_dir)) if f.endswith(".cif")]

    failures = []
    for name, structs in (("canonical mp_20_heldout/train sample", sample), ("canonical CSPBench 180", gt)):
        ok, total = roundtrip(canon, name, structs)
        if ok < 0.995 * total:
            failures.append("%s round-trip %d/%d" % (name, ok, total))
    print("=" * 70)
    for name, structs in (("mp_20_heldout/train", sample[:n_inv]), ("CSPBench 180", gt)):
        rate, fb, err = invariance([("current", current), ("canonical", canon)], name, structs)["canonical"]
        if rate < 0.95 or fb > 0.01 or err:
            failures.append("%s invariance %.3f fallbacks %.3f errors %d" % (name, rate, fb, err))
    if ox == "guess":
        _, bad = sampler_smoke(canon, gt_dir)
        if bad:
            failures.append("sampler smoke %d failures" % bad)

    print("=" * 70)
    if failures:
        print("GATE FAILED:", "; ".join(failures))
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
