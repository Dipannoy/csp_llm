#!/usr/bin/env python3
"""Match rate on the MP-20 test split, per structure, at several budgets.

Not reusable from calculate_success_rate.py, for two reasons. That script keys
predictions by formula with one directory each, but the 9,046 MP-20 test
structures repeat compositions and each must be scored against its own
reference. And it hardcodes its denominator to 180.

Reports both tolerance sets, because they are not interchangeable:

  loose  ltol=0.3, stol=0.5, angle_tol=10   the DiffCSP/CDVAE/CrystaLLM
                                            convention. Required for the
                                            published 51.49% / 77.93% numbers
                                            to mean anything.
  tight  ltol=0.2, stol=0.3, angle_tol=5    pymatgen defaults, and the CSPBench
                                            protocol. Reported alongside so the
                                            two benchmarks stay commensurable.

Parallel over structures with a process pool: StructureMatcher is pure CPU and
the 9,046 comparisons are independent.
"""
import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

TOLERANCES = {
    "loose": dict(ltol=0.3, stol=0.5, angle_tol=10),
    "tight": dict(ltol=0.2, stol=0.3, angle_tol=5),
}


def score_one(task, budgets, gt_dir, pred_dir):
    """Return, per tolerance and budget, whether any of the first N predictions match.

    Candidates are ordered by the integer suffix of their filename, so budget N
    means the first N generated samples -- a name sort would put _10 before _2
    and silently evaluate a different subset.
    """
    key, _formula = task
    gt_path = os.path.join(gt_dir, "%s.cif" % key)
    out = {"key": key, "n_pred": 0, "gt_found": os.path.isfile(gt_path)}
    for tol in TOLERANCES:
        for n in budgets:
            out["%s_at%d" % (tol, n)] = False
    if not out["gt_found"]:
        return out

    try:
        gt = Structure.from_file(gt_path)
    except Exception:
        out["gt_found"] = False
        return out

    files = glob.glob(os.path.join(pred_dir, key, "*.cif"))
    def rank(p):
        stem = os.path.basename(p)[:-4]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 10 ** 9
    files.sort(key=rank)
    out["n_pred"] = len(files)
    if not files:
        return out

    matchers = {t: StructureMatcher(**kw) for t, kw in TOLERANCES.items()}
    biggest = max(budgets)
    # One parse per candidate, reused across tolerances and budgets.
    for idx, path in enumerate(files[:biggest]):
        try:
            pred = Structure.from_file(path)
        except Exception:
            continue
        for tol, m in matchers.items():
            try:
                hit = m.fit(gt, pred)
            except Exception:
                hit = False
            if hit:
                for n in budgets:
                    if idx < n:
                        out["%s_at%d" % (tol, n)] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--budgets", default="1:5:20",
                    help="Comma- or colon-separated. Colons are accepted "
                         "because sbatch --export splits on commas.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model_name", default="model")
    args = ap.parse_args()

    budgets = sorted(int(x) for x in args.budgets.replace(":", ",").split(",") if x)

    targets = []
    with open(args.targets) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                targets.append((parts[0], parts[1]))

    print("%s: %d targets, budgets %s, %d workers"
          % (args.model_name, len(targets), budgets, args.workers), flush=True)

    fn = partial(score_one, budgets=budgets, gt_dir=args.gt_dir,
                 pred_dir=args.pred_dir)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(fn, targets, chunksize=16), 1):
            rows.append(r)
            if i % 1000 == 0:
                print("  %d/%d" % (i, len(targets)), flush=True)

    fields = ["key", "n_pred", "gt_found"] + [
        "%s_at%d" % (t, n) for t in TOLERANCES for n in budgets]
    with open(args.output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # The denominator is every target with a reference structure, so a target
    # that generated nothing scores as a failure instead of vanishing.
    scorable = [r for r in rows if r["gt_found"]]
    n = len(scorable)
    print()
    print("=" * 58)
    print("MP-20 TEST SPLIT -- %s" % args.model_name)
    print("=" * 58)
    print("targets with reference : %d of %d" % (n, len(rows)))
    print("targets with no output : %d" % sum(1 for r in scorable if r["n_pred"] == 0))
    print()
    print("%-7s %s" % ("tol", "  ".join("  @%-3d" % b for b in budgets)))
    for tol in TOLERANCES:
        cells = []
        for b in budgets:
            hits = sum(1 for r in scorable if r["%s_at%d" % (tol, b)])
            cells.append("%6.2f" % (100.0 * hits / n if n else 0.0))
        print("%-7s %s" % (tol, "  ".join(cells)))
    print()
    print("loose = ltol 0.3 / stol 0.5 / angle 10 (DiffCSP convention)")
    print("tight = ltol 0.2 / stol 0.3 / angle 5  (pymatgen defaults, CSPBench)")
    print("saved: %s" % args.output_csv)


if __name__ == "__main__":
    main()
