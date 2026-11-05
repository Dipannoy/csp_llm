"""Independent check: do the 180 scored groundtruth CIFs overlap mp_20_heldout train/val?"""
import os, sys, collections, warnings
import numpy as np
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings("ignore")
root, gt = sys.argv[1], sys.argv[2]
M = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
gts = {}
for f in sorted(os.listdir(gt)):
    if f.endswith(".cif"):
        s = Structure.from_file(os.path.join(gt, f)); s.remove_oxidation_states()
        gts[f[:-4]] = s
keys = {n: s.composition.reduced_composition.formula for n, s in gts.items()}
print("groundtruth CIFs:", len(gts), " distinct reduced compositions:", len(set(keys.values())))
for ds in ("mp_20", "mp_20_heldout"):
    for split in ("train", "val", "test"):
        d = os.path.join(root, ds, split)
        n = np.load(d + "/num_atoms.npy"); z = np.load(d + "/atomic_numbers.npy")
        pos = np.load(d + "/pos.npy"); cell = np.load(d + "/cell.npy")
        off = np.concatenate([[0], np.cumsum(n)])
        bycomp = collections.defaultdict(list)
        for i in range(len(n)):
            c = collections.Counter(int(a) for a in z[off[i]:off[i+1]])
            bycomp[Composition({Element.from_Z(a).symbol: k for a, k in c.items()}).reduced_composition.formula].append(i)
        comp_hits = [g for g, k in keys.items() if k in bycomp]
        struct_hits = []
        for g in comp_hits:
            for i in bycomp[keys[g]]:
                s = Structure(Lattice(cell[i]), z[off[i]:off[i+1]].tolist(), pos[off[i]:off[i+1]], coords_are_cartesian=False)
                if M.fit(gts[g], s):
                    struct_hits.append(g); break
        print("%-14s %-5s %6d structures | benchmark formulas present by composition: %3d | exact structure matches: %3d"
              % (ds, split, len(n), len(comp_hits), len(struct_hits)))
        if ds == "mp_20_heldout" and comp_hits:
            print("    composition hits:", comp_hits[:20])
