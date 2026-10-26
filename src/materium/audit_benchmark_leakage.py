#!/usr/bin/env python3
"""Measure, and optionally remove, benchmark leakage from an MP-20 style split.

The 180 CSP benchmark formulas were never held out of training, so every match
rate measured so far is optimistic by an unknown amount. This reports the overlap
and can write a filtered copy of the dataset with it removed.

Two notions of overlap, reported separately because they mean different things:

  material_id   The benchmark entry itself is in the split. Unambiguous leakage:
                the model was trained on the exact structure it is scored against.

  composition   Some structure in the split has the same reduced composition as a
                benchmark formula -- a different polymorph, or a different cell
                multiple. For this task it is also leakage: the model is given
                only a formula and asked to produce a structure, so having seen
                any structure of that composition is the thing the benchmark is
                supposed to be testing.

Composition is the stricter criterion and the default for removal. It is matched
on the reduced composition so that Zn4S4O16, ZnSO4 and a 2x supercell all collide.

Reads the split's .npy/.json files directly rather than through
MatterGenDataset, so it needs no torch and no GPU.

    # report only
    python audit_benchmark_leakage.py --dataset_root mp_20 \\
        --benchmark_csv csp_materium/CSPbenchmark_test_data.csv

    # write a held-out copy
    python audit_benchmark_leakage.py --dataset_root mp_20 \\
        --benchmark_csv csp_materium/CSPbenchmark_test_data.csv \\
        --write mp_20_heldout
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
from pymatgen.core import Composition, Element

PER_STRUCTURE_HINTS = ("cell.npy", "num_atoms.npy", "structure_id.npy")
PER_ATOM_HINTS = ("pos.npy", "atomic_numbers.npy")


def reduced_key(composition):
    """A hashable reduced composition, so polymorphs and supercells collide."""
    return Composition(composition).reduced_composition.formula


def load_split(path):
    """Read the arrays a split is made of, without constructing a Dataset."""
    need = ("cell.npy", "num_atoms.npy", "pos.npy", "atomic_numbers.npy")
    missing = [f for f in need if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise SystemExit("%s is missing %s" % (path, ", ".join(missing)))

    num_atoms = np.load(os.path.join(path, "num_atoms.npy"))
    atomic_numbers = np.load(os.path.join(path, "atomic_numbers.npy"))

    sid_path = os.path.join(path, "structure_id.npy")
    structure_ids = (
        np.load(sid_path, allow_pickle=True) if os.path.isfile(sid_path) else None
    )

    # Atom arrays are concatenated across structures; these bounds slice them.
    offsets = np.concatenate([[0], np.cumsum(num_atoms)]).astype(np.int64)
    return num_atoms, atomic_numbers, structure_ids, offsets


def split_compositions(num_atoms, atomic_numbers, offsets):
    """Reduced composition of every structure in the split."""
    keys = []
    for i in range(len(num_atoms)):
        zs = atomic_numbers[offsets[i] : offsets[i + 1]]
        counts = Counter(int(z) for z in zs)
        comp = {Element.from_Z(z).symbol: n for z, n in counts.items()}
        keys.append(reduced_key(comp))
    return keys


def classify_files(path, n_structures, n_atoms):
    """Sort a split's files into per-structure, per-atom, and pass-through.

    Done by inspecting leading dimensions rather than from a fixed list, so a
    property file this script has never heard of is still filtered correctly
    instead of being silently copied with the wrong length.
    """
    per_structure, per_atom, verbatim, ambiguous = [], [], [], []

    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue

        if name.endswith(".npy"):
            arr = np.load(full, allow_pickle=True, mmap_mode="r")
            length = arr.shape[0] if arr.ndim else 0
            if name in PER_STRUCTURE_HINTS or length == n_structures:
                per_structure.append(name)
            elif name in PER_ATOM_HINTS or length == n_atoms:
                per_atom.append(name)
            else:
                ambiguous.append((name, length))
        elif name.endswith(".json"):
            try:
                with open(full) as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict) and len(payload.get("values", [])) == n_structures:
                    per_structure.append(name)
                else:
                    verbatim.append(name)
            except Exception:
                verbatim.append(name)
        else:
            verbatim.append(name)

    # n_structures == n_atoms would make the test above ambiguous; it never is
    # for real data (every structure has at least two atoms), but say so rather
    # than silently guessing.
    if n_structures == n_atoms:
        raise SystemExit("Cannot classify files: n_structures == n_atoms.")

    return per_structure, per_atom, verbatim, ambiguous


def write_filtered(src, dst, keep, num_atoms, offsets):
    """Write a copy of `src` into `dst` containing only structures where keep."""
    n_structures = len(num_atoms)
    n_atoms = int(num_atoms.sum())
    per_structure, per_atom, verbatim, ambiguous = classify_files(
        src, n_structures, n_atoms
    )
    if ambiguous:
        raise SystemExit(
            "Unclassifiable files in %s (length matches neither structures nor "
            "atoms): %s" % (src, ambiguous)
        )

    os.makedirs(dst, exist_ok=True)

    # Atom-level mask built from the per-structure mask via the same offsets the
    # dataset uses, so the atom arrays stay aligned with num_atoms.
    atom_keep = np.zeros(n_atoms, dtype=bool)
    for i in np.flatnonzero(keep):
        atom_keep[offsets[i] : offsets[i + 1]] = True

    for name in per_structure:
        if name.endswith(".npy"):
            arr = np.load(os.path.join(src, name), allow_pickle=True)
            np.save(os.path.join(dst, name), arr[keep])
        else:
            with open(os.path.join(src, name)) as fh:
                payload = json.load(fh)
            payload["values"] = [v for v, k in zip(payload["values"], keep) if k]
            with open(os.path.join(dst, name), "w") as fh:
                json.dump(payload, fh)

    for name in per_atom:
        arr = np.load(os.path.join(src, name), allow_pickle=True)
        np.save(os.path.join(dst, name), arr[atom_keep])

    for name in verbatim:
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    return {
        "per_structure": per_structure,
        "per_atom": per_atom,
        "verbatim": verbatim,
        "kept_structures": int(keep.sum()),
        "kept_atoms": int(atom_keep.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="Directory holding the split subdirectories, e.g. mp_20")
    ap.add_argument("--benchmark_csv", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--write", default=None, metavar="OUT_ROOT",
                    help="Write a filtered copy here. Without it, only report.")
    ap.add_argument("--by", choices=["composition", "material_id"],
                    default="composition",
                    help="Criterion for removal (default: composition, the stricter).")
    args = ap.parse_args()

    with open(args.benchmark_csv) as fh:
        rows = list(csv.DictReader(fh))
    bench_ids = {r["material_id"] for r in rows}
    bench_comps = {}
    for r in rows:
        bench_comps.setdefault(reduced_key(r["primitive_formula"]), []).append(
            r["primitive_formula"]
        )

    print("benchmark: %d entries, %d distinct reduced compositions"
          % (len(rows), len(bench_comps)))
    print("removal criterion: %s" % args.by)
    print()

    hit_comps_overall = set()
    hit_ids_overall = set()
    totals = Counter()

    for split in args.splits.split(","):
        src = os.path.join(args.dataset_root, split)
        if not os.path.isdir(src):
            print("%-6s  (absent)" % split)
            continue

        num_atoms, atomic_numbers, structure_ids, offsets = load_split(src)
        comps = split_compositions(num_atoms, atomic_numbers, offsets)

        comp_hit = np.array([c in bench_comps for c in comps])
        if structure_ids is not None:
            ids = [str(s) for s in structure_ids]
            id_hit = np.array([i in bench_ids for i in ids])
        else:
            ids = None
            id_hit = np.zeros(len(comps), dtype=bool)

        remove = id_hit if args.by == "material_id" else comp_hit
        keep = ~remove

        print("%-6s  %6d structures" % (split, len(comps)))
        print("          %6d match by material_id%s"
              % (id_hit.sum(), "" if ids else "   (no structure_id.npy)"))
        print("          %6d match by composition  (%d distinct benchmark formulas)"
              % (comp_hit.sum(), len({c for c in comps if c in bench_comps})))
        print("          %6d would be removed, %d kept"
              % (remove.sum(), keep.sum()))

        totals[split + "_total"] = len(comps)
        totals[split + "_removed"] = int(remove.sum())
        hit_comps_overall |= {c for c in comps if c in bench_comps}
        if ids:
            hit_ids_overall |= {i for i in ids if i in bench_ids}

        if args.write:
            dst = os.path.join(args.write, split)
            info = write_filtered(src, dst, keep, num_atoms, offsets)
            print("          wrote %s: %d structures, %d atoms"
                  % (dst, info["kept_structures"], info["kept_atoms"]))
            print("          per-structure: %s" % ", ".join(info["per_structure"]))
            print("          per-atom     : %s" % ", ".join(info["per_atom"]))
            if info["verbatim"]:
                print("          copied as-is : %s" % ", ".join(info["verbatim"]))
        print()

    print("=" * 66)
    print("benchmark coverage by the dataset")
    print("=" * 66)
    print("  %d of %d benchmark material_ids appear in the splits"
          % (len(hit_ids_overall), len(bench_ids)))
    print("  %d of %d benchmark compositions appear in the splits"
          % (len(hit_comps_overall), len(bench_comps)))
    print()

    missed = set(bench_comps) - hit_comps_overall
    if missed:
        print("  compositions NOT in the dataset (%d):" % len(missed))
        for key in sorted(missed)[:10]:
            print("      %s" % ", ".join(bench_comps[key]))
        if len(missed) > 10:
            print("      ... and %d more" % (len(missed) - 10))
        print()

    if not args.write:
        print("Nothing was written. Re-run with --write <OUT_ROOT> to create the")
        print("filtered copy, then train with --dataset <OUT_ROOT>. Use a new name:")
        print("the token cache key includes the dataset name, so a fresh name")
        print("forces re-tokenization instead of reusing the leaked cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
