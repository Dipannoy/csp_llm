#!/usr/bin/env python3
"""Build a top-N view of a relaxed run, for match-rate-at-N evaluation.

The relaxation step writes `<FORMULA>_<rank>.cif`, where rank 1 is the lowest
formation energy. A top-N view is a directory holding symlinks to ranks 1..N
only, so calculate_success_rate.py scores exactly the N most stable candidates
without needing a --top_n flag of its own.

Symlinks rather than copies: the views for N = 1, 5, 10, 25 together would
otherwise duplicate the relaxed set four times over.

Every formula in the groundtruth gets a directory, even when it has no
structures. calculate_success_rate.py hardcodes its denominator to 180, so an
absent formula does not shift the rate -- but it would vanish from the per-formula
CSV, which is where a missing formula has to stay visible.

Only the standard library is used, so this runs in whichever environment the
aggregation stage activates.
"""
import argparse
import os
import re
import sys


def ranked_cifs(formula_dir, formula):
    """Return this formula's CIFs ordered by saved rank, lowest energy first.

    Sorted numerically on the rank field: a lexicographic sort would place
    rank 10 before rank 2 and quietly build the wrong top-5.
    """
    pattern = re.compile(r'^' + re.escape(formula) + r'_(\d+)\.cif$')
    ranked = []
    try:
        names = os.listdir(formula_dir)
    except OSError:
        return []

    for name in names:
        m = pattern.match(name)
        if m:
            ranked.append((int(m.group(1)), name))

    # Fall back to name order for any file not following <formula>_<rank>.cif,
    # so a differently named output set is still evaluated rather than skipped.
    if not ranked:
        return sorted(n for n in names if n.endswith('.cif'))

    ranked.sort()
    return [name for _rank, name in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax_root", required=True)
    ap.add_argument("--view_root", required=True,
                    help="Directory to create; filled with one subdir per formula.")
    ap.add_argument("--top_n", type=int, required=True)
    ap.add_argument("--groundtruth_dir", required=True)
    args = ap.parse_args()

    formulas = sorted(f[:-4] for f in os.listdir(args.groundtruth_dir)
                      if f.endswith(".cif"))
    if not formulas:
        sys.exit("No .cif files in %s" % args.groundtruth_dir)

    os.makedirs(args.view_root, exist_ok=True)

    linked = 0
    with_any = 0
    short = []
    for formula in formulas:
        src_dir = os.path.join(args.relax_root, formula)
        dst_dir = os.path.join(args.view_root, formula)
        os.makedirs(dst_dir, exist_ok=True)

        # Rebuilt from scratch each run: a view left over from an earlier,
        # smaller relaxation would otherwise keep scoring stale structures.
        for stale in os.listdir(dst_dir):
            os.unlink(os.path.join(dst_dir, stale))

        names = ranked_cifs(src_dir, formula)[:args.top_n]
        if names:
            with_any += 1
        if 0 < len(names) < args.top_n:
            short.append((formula, len(names)))

        for name in names:
            os.symlink(os.path.abspath(os.path.join(src_dir, name)),
                       os.path.join(dst_dir, name))
            linked += 1

    print("top-%d view: %s" % (args.top_n, args.view_root))
    print("  formulas            : %d" % len(formulas))
    print("  with >=1 structure  : %d" % with_any)
    print("  symlinks created    : %d" % linked)
    if short:
        # Not an error: a formula with fewer than N relaxed structures is scored
        # on everything it has. Reported so a top-25 rate built mostly from
        # formulas holding 10 structures is not mistaken for a true top-25.
        print("  fewer than %d available: %d formulas" % (args.top_n, len(short)))
        for formula, n in short[:10]:
            print("      %-16s %d" % (formula, n))
        if len(short) > 10:
            print("      ... and %d more" % (len(short) - 10))


if __name__ == "__main__":
    main()
