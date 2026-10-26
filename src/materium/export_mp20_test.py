#!/usr/bin/env python3
"""Export the MP-20 test split as ground truth for the standard CSP protocol.

The CSPBench driver keys predictions by formula, one directory per formula. That
cannot work here: the 9,046 test structures contain the same composition many
times over, and the standard MP-20 task scores each structure against its own
reference. So the key is the material_id, and the formula travels beside it.

Writes:
  <out>/cifs/<material_id>.cif    the reference structure
  <out>/targets.tsv              material_id, formula, nsites, spacegroup
"""
import json, os, sys
from collections import Counter

import numpy as np
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SYMPREC = 0.1

def main(split_dir, out_dir):
    num = np.load(os.path.join(split_dir, "num_atoms.npy"))
    cell = np.load(os.path.join(split_dir, "cell.npy"))
    pos = np.load(os.path.join(split_dir, "pos.npy"))
    z = np.load(os.path.join(split_dir, "atomic_numbers.npy"))
    sid = np.load(os.path.join(split_dir, "structure_id.npy"), allow_pickle=True)
    off = np.concatenate([[0], np.cumsum(num)]).astype(np.int64)

    cif_dir = os.path.join(out_dir, "cifs")
    os.makedirs(cif_dir, exist_ok=True)

    rows, failed = [], 0
    for i in range(len(num)):
        mid = str(sid[i])
        try:
            sp = [Element.from_Z(int(x)).symbol for x in z[off[i]:off[i+1]]]
            # pos.npy is FRACTIONAL. The MatterGenDataset docstring says
            # Cartesian and is wrong; datasets.py uses coords_are_cartesian=False.
            st = Structure(Lattice(cell[i]), sp, pos[off[i]:off[i+1]],
                           coords_are_cartesian=False)
            st.to(filename=os.path.join(cif_dir, "%s.cif" % mid), fmt="cif")
            try:
                sg = SpacegroupAnalyzer(st, symprec=SYMPREC).get_space_group_symbol()
            except Exception:
                sg = "P1"
            # The formula the generator will be conditioned on: the integer
            # composition of this cell, not its reduced form, so the atom count
            # the model must produce matches the reference.
            counts = Counter(sp)
            formula = "".join("%s%d" % (el, counts[el]) for el in sorted(counts))
            rows.append((mid, formula, len(st), sg))
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print("  [skip] %s: %s" % (mid, exc))

        if (i + 1) % 2000 == 0:
            print("  %d/%d" % (i + 1, len(num)), flush=True)

    with open(os.path.join(out_dir, "targets.tsv"), "w") as fh:
        for r in rows:
            fh.write("%s\t%s\t%d\t%s\n" % r)

    sizes = Counter(r[2] for r in rows)
    print()
    print("exported %d structures (%d failed) to %s" % (len(rows), failed, out_dir))
    print("distinct formulas: %d of %d structures"
          % (len({r[1] for r in rows}), len(rows)))
    print("atom counts: min=%d max=%d" % (min(sizes), max(sizes)))
    print("P1 fraction: %.3f" % (sum(1 for r in rows if r[3] == "P1") / len(rows)))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
