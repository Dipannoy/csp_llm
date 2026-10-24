#!/usr/bin/env python3
"""Check which symmetries of a crystal the tokenizer actually respects.

A crystal is a periodic set with no intrinsic ordering, origin, or choice of
cell. Any of those conventions that survives into the token stream is capacity
the model spends memorising bookkeeping instead of physics.

Four checks, each: transform a structure, re-tokenize, diff the token lists.

  1. SITE ORDER        shuffle the sites.
                       Expected to pass -- tokenize() sorts them.

  2. LATTICE VECTORS   permute a, b, c and the matching coordinate components.
                       Expected to pass -- tokenize() Niggli-reduces first.
                       Worth checking anyway: Niggli reduction is numerically
                       fragile for near-degenerate cells, and a silent failure
                       means cell conventions leak into the tokens.

  3. ORIGIN            shift every fractional coordinate by the same delta.
                       Expected to FAIL, by design: this codebase handles the
                       origin degeneracy by augmenting translations during
                       training rather than by canonicalising the origin. The
                       check is here to quantify the orbit, not to pass.

  4. AUGMENTATION      the one that matters. Shift the origin by an exact
                       multiple of the bin width, then compare the re-tokenized
                       sequence against what apply_translation_augmentation
                       produces from the original tokens for that same shift.

                       These must agree. The augmentation only rewrites
                       coordinate bins arithmetically, bin -> (bin + k) mod B,
                       whereas a real origin shift also changes the
                       frac_coords tie-break that sorts same-species atoms
                       (tokenizer.py, the sorted(...) call in tokenize), so an
                       actual shift can REORDER the atom blocks. If they
                       disagree, training has been fed token sequences the
                       tokenizer would never produce from any real crystal.

Usage:
    python check_tokenizer_invariance.py --ckpt <path/to/model.pt> \\
        --cif_dir /work/dg47/MLEG/CSPBenchMetrics/data/180-primitive-groundtruth

Needs pymatgen and torch, so run it in llm_env. No GPU.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from materium.coord_loss import (
    ROLE_COORD,
    TokenLayout,
    build_token_roles,
    resort_atom_blocks,
)
from materium.tokenizer import CrystalTokenizer, SequenceOrder, SortingOrder


# --------------------------------------------------------------------- loading

def resolve_checkpoint(ckpt):
    """Accept either a usable path or the <dir>/<file> form llm_generate.py uses.

    llm_generate.py resolves checkpoints against src/materium/checkpoints/ rather
    than the working directory, so the same string that works for generation is
    not a valid path from the repo root. Try it as given first, then there.
    """
    candidates = [ckpt, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "checkpoints", ckpt)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise SystemExit(
        "Checkpoint not found. Tried:\n"
        + "".join("  %s\n" % c for c in candidates)
        + "Pass either a full path, or the <ckpt_dir>/<ckpt_file> that\n"
          "llm_generate.py takes (resolved under src/materium/checkpoints/)."
    )


def load_tokenizer(ckpt):
    """Recover the tokenizer a checkpoint was trained with.

    Mirrors llm_generate.py: the config travels inside the checkpoint, and only
    a checkpoint written before that was stored falls back to defaults.
    """
    if ckpt:
        ckpt_path = resolve_checkpoint(ckpt)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = state.get("tokenizer")
        if cfg is not None:
            print("tokenizer: loaded from %s" % ckpt_path)
            return CrystalTokenizer.from_dict(cfg)
        print("tokenizer: checkpoint has no config, using defaults")

    from pymatgen.core import Element
    print("tokenizer: defaults (ELEMENT_VOCAB 1-99, 1024 bins, SPECIES/ATOMS_FIRST)")
    return CrystalTokenizer(
        element_vocab=[Element.from_Z(i).symbol for i in range(1, 100)],
        lattice_stats={
            "a": (2.0, 10.0), "b": (2.0, 12.5), "c": (2.0, 20.0),
            "alpha": (60.0, 120.0), "beta": (60.0, 120.0), "gamma": (60.0, 120.0),
        },
        num_quant_bins=1024,
        sorting_order=SortingOrder.SPECIES,
        sequence_order=SequenceOrder.ATOMS_FIRST,
    )


# ------------------------------------------------------------- transformations

def shuffle_sites(structure, rng):
    idx = list(range(len(structure)))
    rng.shuffle(idx)
    return Structure(
        lattice=structure.lattice,
        species=[structure[i].species for i in idx],
        coords=[structure[i].frac_coords for i in idx],
        coords_are_cartesian=False,
    )


def permute_lattice_vectors(structure, perm):
    """Relabel the cell axes, carrying the coordinate components with them.

    Permuting the rows of the lattice matrix and applying the same permutation
    to each fractional coordinate leaves every Cartesian position, and hence the
    crystal, unchanged -- only the choice of which axis is called 'a' differs.
    """
    matrix = structure.lattice.matrix[list(perm), :]
    return Structure(
        lattice=Lattice(matrix),
        species=[s.species for s in structure],
        coords=[s.frac_coords[list(perm)] for s in structure],
        coords_are_cartesian=False,
    )


def shift_origin(structure, delta):
    return Structure(
        lattice=structure.lattice,
        species=[s.species for s in structure],
        coords=[(s.frac_coords + delta) % 1.0 for s in structure],
        coords_are_cartesian=False,
    )


# ------------------------------------------------------- augmentation equivalent

def shift_coord_bins(tokens, layout, shifts):
    """Apply a known per-axis bin shift, the way the training augmentation does.

    This is apply_translation_augmentation with the random draw replaced by a
    fixed `shifts`, so the comparison in check 4 is against a specific origin
    shift rather than an arbitrary one. Everything else -- role and axis
    derivation via build_token_roles, and the re-sort via resort_atom_blocks --
    is the production code, which is what check 4 is testing.
    """
    tokens = torch.as_tensor(tokens, dtype=torch.long).unsqueeze(0)
    roles, coord_axis = build_token_roles(tokens, layout)

    is_coord = roles == ROLE_COORD
    axis = coord_axis.clamp_min(0)

    per_token_shift = torch.as_tensor(shifts, dtype=torch.long)[axis]
    bins = tokens - layout.quant_offset
    shifted = (bins + per_token_shift) % layout.num_quant_bins + layout.quant_offset

    shifted_tokens = torch.where(is_coord, shifted, tokens)
    return resort_atom_blocks(shifted_tokens, roles, layout).squeeze(0).tolist()


# -------------------------------------------------------------------- the checks

def first_difference(a, b):
    """Describe how two token lists diverge, for reporting."""
    if len(a) != len(b):
        return "length %d vs %d" % (len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "first differs at position %d: %d vs %d (%d of %d positions differ)" % (
                i, x, y, sum(1 for p, q in zip(a, b) if p != q), len(a))
    return None


def run(tokenizer, structures, rng, seed):
    layout = tokenizer.token_layout()
    nbins = tokenizer.num_quant_bins

    results = {k: {"pass": 0, "fail": 0, "error": 0, "examples": []}
               for k in ("site_order", "lattice_vectors", "origin", "augmentation")}

    for name, structure in structures:
        try:
            base = tokenizer.tokenize(structure)
        except Exception as exc:
            for k in results:
                results[k]["error"] += 1
            print("  [error] %s: could not tokenize: %s" % (name, exc))
            continue

        trials = []

        # 1. site order
        trials.append(("site_order", shuffle_sites(structure, rng), None))

        # 2. lattice vectors: a cyclic relabelling of the axes
        trials.append(("lattice_vectors", permute_lattice_vectors(structure, (1, 2, 0)), None))

        # 3. origin: an arbitrary shift, not aligned to the bin grid
        delta = np.array([rng.random(), rng.random(), rng.random()])
        trials.append(("origin", shift_origin(structure, delta), None))

        # 4. origin shift by an exact multiple of the bin width, compared against
        #    the augmentation rather than against the unshifted tokens
        shifts = [rng.randrange(nbins) for _ in range(3)]
        k_delta = np.array(shifts, dtype=float) / nbins
        trials.append(("augmentation", shift_origin(structure, k_delta), shifts))

        for key, transformed, shifts_used in trials:
            try:
                got = tokenizer.tokenize(transformed)
                want = base if shifts_used is None else shift_coord_bins(base, layout, shifts_used)
            except Exception as exc:
                results[key]["error"] += 1
                continue

            diff = first_difference(want, got)
            if diff is None:
                results[key]["pass"] += 1
            else:
                results[key]["fail"] += 1
                if len(results[key]["examples"]) < 3:
                    results[key]["examples"].append((name, diff))

    return results


LABELS = {
    "site_order": ("1. site order (shuffle sites)", "pass"),
    "lattice_vectors": ("2. lattice vectors (permute a,b,c)", "pass"),
    "origin": ("3. origin (arbitrary shift)", "fail"),
    "augmentation": ("4. augmentation vs real origin shift", "pass"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint whose tokenizer config to use. Without it, "
                         "the llm_generate.py defaults are used.")
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--num", type=int, default=30,
                    help="How many structures to test (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tokenizer = load_tokenizer(args.ckpt)

    names = sorted(f for f in os.listdir(args.cif_dir) if f.endswith(".cif"))
    if args.num:
        names = names[:args.num]

    structures = []
    for n in names:
        try:
            structures.append((n[:-4], Structure.from_file(os.path.join(args.cif_dir, n))))
        except Exception as exc:
            print("  [skip] %s: %s" % (n, exc))

    print("structures: %d from %s" % (len(structures), args.cif_dir))
    print("bins      : %d" % tokenizer.num_quant_bins)
    print("sorting   : %s | sequence: %s"
          % (tokenizer.sorting_order.name, tokenizer.sequence_order.name))
    print()

    results = run(tokenizer, structures, rng, args.seed)

    print("=" * 70)
    print("TOKENIZER INVARIANCE")
    print("=" * 70)
    worrying = []
    for key, (label, expected) in LABELS.items():
        r = results[key]
        total = r["pass"] + r["fail"]
        rate = (100.0 * r["pass"] / total) if total else 0.0
        verdict = "pass" if r["fail"] == 0 else "fail"
        flag = "" if verdict == expected else "   <-- UNEXPECTED"
        print("%-40s %3d/%-3d invariant (%5.1f%%)  expected %s%s"
              % (label, r["pass"], total, rate, expected, flag))
        if r["error"]:
            print("%-40s %3d errored" % ("", r["error"]))
        if verdict != expected:
            worrying.append((key, label, r))

    print()
    for key, (label, expected) in LABELS.items():
        r = results[key]
        if r["examples"] and (expected == "pass"):
            print("%s -- examples:" % label)
            for name, diff in r["examples"]:
                print("    %-16s %s" % (name, diff))
            print()

    print("-" * 70)
    if not worrying:
        print("Everything behaved as expected.")
        print()
        print("Check 3 failing is the intended design: the origin degeneracy is")
        print("handled by training-time augmentation, not by canonicalisation.")
        print("Check 4 passing means that augmentation agrees with what the")
        print("tokenizer produces for a real origin shift.")
        return 0

    print("UNEXPECTED RESULTS")
    for key, label, r in worrying:
        print()
        print("  %s" % label)
        if key == "augmentation":
            print("    The training augmentation does not reproduce what the")
            print("    tokenizer emits for the same origin shift. The augmentation")
            print("    only rewrites coordinate bins; a real shift also changes the")
            print("    frac_coords tie-break that orders same-species atoms, so the")
            print("    atom blocks get permuted as well. Every augmented example")
            print("    has been a sequence no real crystal would tokenize to.")
        elif key == "lattice_vectors":
            print("    Niggli reduction did not canonicalise the cell, so which")
            print("    axis is called 'a' leaks into the tokens and the model must")
            print("    memorise cell conventions.")
        elif key == "site_order":
            print("    Site sorting is not deterministic, so the same crystal has")
            print("    several token sequences for no physical reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
