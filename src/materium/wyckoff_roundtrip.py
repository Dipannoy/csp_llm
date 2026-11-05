#!/usr/bin/env python3
"""Gate for the Wyckoff representation: does it reproduce its own input?

No model is trained on a representation until this passes. The legacy bin grid
round-tripped to 1.5 bin widths and quietly capped every result built on it;
the same check here, before any training, is what prevents a repeat.

For each structure: encode to Wyckoff form, decode back, and score the result
against the original with the benchmark's own StructureMatcher settings and
space-group tolerance. Done twice -- with exact free parameters, and with free
parameters and lattice quantised to the 1024-bin grid the model will predict
on -- so the cost of quantisation is measured separately from the cost of the
representation itself.

Also reports the statistics the tokenizer design depends on: orbits against
atoms, free parameters against 3N, the lattice ranges of the standard-setting
cell, and how many distinct Wyckoff positions actually occur.

    python wyckoff_roundtrip.py <mp20 split dir> <groundtruth cif dir> [n_train]
"""
import collections
import os
import random
import sys
import time

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from materium.wyckoff import SYMPREC, WyckoffCrystal, WyckoffSite, decode, encode

BINS = 1024
MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)


def load_split(d):
    pos = np.load(os.path.join(d, "pos.npy"))
    cell = np.load(os.path.join(d, "cell.npy"))
    z = np.load(os.path.join(d, "atomic_numbers.npy"))
    n = np.load(os.path.join(d, "num_atoms.npy"))
    off = np.concatenate([[0], np.cumsum(n)])
    for i in range(len(n)):
        a, b = off[i], off[i + 1]
        # pos.npy is FRACTIONAL, whatever the MatterGenDataset docstring says;
        # read as Cartesian, every structure is scrambled to near-P1 and the
        # round-trip is measured on noise.
        yield Structure(Lattice(cell[i]), z[a:b].tolist(), pos[a:b],
                        coords_are_cartesian=False)


def quantise(wc: WyckoffCrystal, lat_range) -> WyckoffCrystal:
    """Free parameters and lattice through the 1024-bin grid and back."""
    q = lambda v: (min(int(np.floor(v * BINS)), BINS - 1) + 0.5) / BINS

    def ql(v, lo, hi):
        t = (np.clip(v, lo, hi) - lo) / (hi - lo)
        return lo + q(t) * (hi - lo)

    lat = tuple(ql(v, lo, hi) for v, (lo, hi) in zip(wc.lattice, lat_range))
    sites = [WyckoffSite(s.element, s.wp_index, s.label, s.multiplicity,
                         [q(v % 1.0) for v in s.free]) for s in wc.sites]
    return WyckoffCrystal(wc.space_group, lat, sites)


def check(struct, wc, label_counts):
    """(exact match, exact sg ok, decoded) for one structure."""
    dec = decode(wc)
    # The representation stores elements only (oxidation is a separate token),
    # and 7 of the groundtruth CIFs are decorated with charges such as Sb0.83-.
    # StructureMatcher's default comparator demands identical species, so it
    # would fail those on charge alone, whatever the geometry.
    ref = struct.copy()
    ref.remove_oxidation_states()
    ok = MATCHER.fit(ref, dec)
    sg = SpacegroupAnalyzer(dec, symprec=SYMPREC).get_space_group_number()
    return ok, sg == wc.space_group, dec


def run(name, structs, lat_range=None):
    t0 = time.time()
    n = enc_fail = dec_fail = 0
    exact_ok = exact_sg = q_ok = q_sg = 0
    orbits = atoms = free = 0
    labels = collections.Counter()
    lat_vals = []
    fails = []

    encoded = []
    for s in structs:
        n += 1
        try:
            wc = encode(s)
        except Exception as e:
            enc_fail += 1
            fails.append(("encode", s.composition.reduced_formula, repr(e)[:120]))
            continue
        encoded.append((s, wc))
        orbits += len(wc.sites)
        atoms += wc.num_atoms
        free += wc.num_free
        lat_vals.append(wc.lattice)
        for site in wc.sites:
            labels[(wc.space_group, site.wp_index)] += 1

    lat_vals = np.array(lat_vals)
    if not len(lat_vals):
        print("%s: all %d encodes failed" % (name, n))
        for f in fails[:12]:
            print("    fail:", f)
        sys.exit(1)
    if lat_range is None:
        # Grid range for the quantised check, with a margin. Reported, because
        # the tokenizer will have to fix it.
        lo, hi = lat_vals.min(0), lat_vals.max(0)
        lat_range = [(float(l) * 0.95, float(h) * 1.05) for l, h in zip(lo, hi)]

    for s, wc in encoded:
        try:
            ok, sgok, _ = check(s, wc, labels)
            exact_ok += ok
            exact_sg += sgok
            ok, sgok, _ = check(s, quantise(wc, lat_range), labels)
            q_ok += ok
            q_sg += sgok
            if not ok:
                fails.append(("match", s.composition.reduced_formula, "sg %d" % wc.space_group))
        except Exception as e:
            dec_fail += 1
            fails.append(("decode", s.composition.reduced_formula, repr(e)[:120]))

    pct = lambda k: 100.0 * k / max(n, 1)
    print("=" * 70)
    print("%s: %d structures, %.0f s" % (name, n, time.time() - t0))
    print("  encode failures       %5d  (%.2f%%)" % (enc_fail, pct(enc_fail)))
    print("  decode failures       %5d  (%.2f%%)" % (dec_fail, pct(dec_fail)))
    print("  exact    structure    %5d  (%.2f%%)   space group %5d (%.2f%%)"
          % (exact_ok, pct(exact_ok), exact_sg, pct(exact_sg)))
    print("  1024-bin structure    %5d  (%.2f%%)   space group %5d (%.2f%%)"
          % (q_ok, pct(q_ok), q_sg, pct(q_sg)))
    if encoded:
        k = len(encoded)
        print("  per structure: %.1f orbits for %.1f atoms (standard cell); "
              "%.1f free params vs %.1f coordinates in the current tokenizer"
              % (orbits / k, atoms / k, free / k, 3.0 * atoms / k))
        print("  distinct (group, Wyckoff) tokens: %d" % len(labels))
        print("  lattice min  %s" % np.round(lat_vals.min(0), 2))
        print("  lattice max  %s" % np.round(lat_vals.max(0), 2))
    for f in fails[:12]:
        print("    fail:", f)
    return lat_range


def main():
    split, gt_dir = sys.argv[1], sys.argv[2]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    random.seed(0)
    train = list(load_split(split))
    sample = random.sample(train, min(n_train, len(train)))
    lat_range = run("mp_20_heldout/train sample", sample)

    gt = [Structure.from_file(os.path.join(gt_dir, f))
          for f in sorted(os.listdir(gt_dir)) if f.endswith(".cif")]
    # The benchmark is quantised on the training range, as the model would be.
    run("CSPBench 180 groundtruth", gt, lat_range)


if __name__ == "__main__":
    main()
