#!/usr/bin/env python3
"""Reconcile a pipeline run against the benchmark, and prepare it for matching.

Three jobs, all about not losing formulas silently:

1.  Report every benchmark formula's fate per stage, from the state markers, so
    a failure is attributable to generation or relaxation rather than appearing
    as an absence.

2.  Create an empty output directory for any formula with no relaxed structures.
    calculate_success_rate.py builds total_formulas from os.scandir(pred_base_dir),
    counting only the directories it finds, so a formula with no directory drops
    out of the totals entirely -- that is how a 180-formula benchmark reported 175
    and a 58.86% rate that was really 57.22%. An empty directory makes the formula
    count as the failure it is.

3.  Exit non-zero if any formula is incomplete, so the caller can choose to
    resubmit before spending time on matching.

Uses only the standard library, so it runs in whichever environment the
aggregation stage happens to activate.
"""
import argparse
import csv
import glob
import os
import sys


def read_marker(path):
    """Marker contents are a single `k=v k=v` line; absent file returns None."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def marker_field(text, key):
    if not text:
        return ""
    for part in text.split():
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth_dir", required=True,
                    help="The 180 <formula>.cif reference structures. Used as the "
                         "formula list because calculate_success_rate.py looks up "
                         "every prediction by exactly these names.")
    ap.add_argument("--state_dir", required=True)
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--relax_root", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--fill_empty", action="store_true",
                    help="Create an empty relaxed dir for formulas that have none, "
                         "so the matching denominator covers the whole benchmark.")
    args = ap.parse_args()

    formulas = sorted(f[:-4] for f in os.listdir(args.groundtruth_dir)
                      if f.endswith(".cif"))
    if not formulas:
        sys.exit("No .cif files in %s" % args.groundtruth_dir)

    rows = []
    for formula in formulas:
        state = os.path.join(args.state_dir, formula)
        gen_ok = read_marker(os.path.join(state, "generate.ok"))
        gen_fail = read_marker(os.path.join(state, "generate.fail"))
        rel_ok = read_marker(os.path.join(state, "relax.ok"))
        rel_fail = read_marker(os.path.join(state, "relax.fail"))

        gen_dir = os.path.join(args.gen_root, "materium_inference_" + formula)
        relax_dir = os.path.join(args.relax_root, formula)

        # Counted from disk rather than trusted from the marker, so a file
        # removed after the fact cannot leave a stale "ok".
        n_gen = len(glob.glob(os.path.join(gen_dir, "*", "*.cif")))
        n_relax = len(glob.glob(os.path.join(relax_dir, "*.cif")))

        if rel_ok and n_relax:
            stage, status = "relax", "ok"
        elif rel_fail:
            stage = "relax"
            status = marker_field(rel_fail, "reason") or "failed"
        elif gen_fail:
            stage = "generate"
            status = marker_field(gen_fail, "reason") or "failed"
        elif gen_ok:
            stage, status = "relax", "not_started"
        else:
            stage, status = "generate", "not_started"

        rows.append({
            "formula": formula,
            "stage": stage,
            "status": status,
            "n_generated": n_gen,
            "n_relaxed": n_relax,
            "generate_marker": gen_ok or gen_fail or "",
            "relax_marker": rel_ok or rel_fail or "",
        })

    with open(args.manifest_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    complete = [r for r in rows if r["status"] == "ok"]
    incomplete = [r for r in rows if r["status"] != "ok"]

    print("=" * 66)
    print("PIPELINE STATUS")
    print("=" * 66)
    print("benchmark formulas : %d" % len(rows))
    print("relaxed structures : %d" % len(complete))
    print("incomplete         : %d" % len(incomplete))
    print()

    by_status = {}
    for r in incomplete:
        by_status.setdefault((r["stage"], r["status"]), []).append(r["formula"])
    if by_status:
        print("incomplete, grouped by where they stopped:")
        for (stage, status), names in sorted(by_status.items()):
            print("  %-9s %-24s %3d" % (stage, status, len(names)))
            for n in sorted(names)[:8]:
                print("      %s" % n)
            if len(names) > 8:
                print("      ... and %d more" % (len(names) - 8))
        print()

    filled = 0
    if args.fill_empty:
        for r in rows:
            if r["n_relaxed"] == 0:
                os.makedirs(os.path.join(args.relax_root, r["formula"]), exist_ok=True)
                filled += 1
        print("created %d empty output dirs so the matching denominator is %d, "
              "not %d" % (filled, len(rows), len(complete)))
        print()

    print("manifest: %s" % args.manifest_csv)

    # Non-zero tells submit.sh/run_aggregate.sh that a resubmit would still find
    # work to do. Matching can still proceed -- the empty dirs make the result
    # honest -- so this is advisory, not fatal, for the caller.
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
