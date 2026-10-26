#!/usr/bin/env python3
"""Write space_group.json for an MP-20 style dataset, via spglib.

The dataset's optional space_group.json is absent, so MatterGenDataset falls back
to "P1" for every structure and marks it unknown. Any objective that uses the
space group -- an auxiliary prediction head, or space-group conditioning -- would
train on that constant and learn nothing. This derives the real symbol from the
stored cell and positions.

Writes {"values": [symbol, ...]} with one entry per structure, matching the
format _load_optional_json expects.
"""
import json, os, sys
import numpy as np
from pymatgen.core import Lattice, Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SYMPREC = 0.1   # matches the tolerance the match-rate script assigns with

def main(root, splits):
    for split in splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            print("%s absent, skipping" % d)
            continue

        num = np.load(os.path.join(d, "num_atoms.npy"))
        cell = np.load(os.path.join(d, "cell.npy"))
        pos = np.load(os.path.join(d, "pos.npy"))
        z = np.load(os.path.join(d, "atomic_numbers.npy"))
        off = np.concatenate([[0], np.cumsum(num)]).astype(np.int64)

        symbols, failed = [], 0
        for i in range(len(num)):
            try:
                lat = Lattice(cell[i])
                sp = [Element.from_Z(int(x)).symbol for x in z[off[i]:off[i+1]]]
                # pos.npy holds FRACTIONAL coordinates. The MatterGenDataset
                # docstring says Cartesian, but datasets.py builds its Structure
                # with coords_are_cartesian=False, which is what the data is.
                # Reading them as Cartesian collapses every cell and spglib then
                # reports P1 for three quarters of the dataset.
                st = Structure(lat, sp, pos[off[i]:off[i+1]], coords_are_cartesian=False)
                symbols.append(SpacegroupAnalyzer(st, symprec=SYMPREC).get_space_group_symbol())
            except Exception:
                failed += 1
                symbols.append("P1")   # the same fallback the loader uses

            if (i + 1) % 5000 == 0:
                print("  %s %d/%d" % (split, i + 1, len(num)), flush=True)

        out = os.path.join(d, "space_group.json")
        with open(out, "w") as fh:
            json.dump({"values": symbols}, fh)

        from collections import Counter
        top = Counter(symbols).most_common(8)
        print("%s: %d structures, %d failed -> %s" % (split, len(symbols), failed, out))
        print("   P1 fraction: %.3f" % (symbols.count("P1") / len(symbols)))
        print("   most common: %s" % ", ".join("%s(%d)" % t for t in top))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["train", "val", "test"])
