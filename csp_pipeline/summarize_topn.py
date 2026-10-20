#!/usr/bin/env python3
"""Collect the top-N match rates of one run into a single table.

Rates are parsed from each top-N run's captured stdout rather than recomputed
from the per-formula CSVs, so the numbers reported here are exactly the ones
calculate_success_rate.py produced -- including its hardcoded denominator of 180.

Writes a tidy CSV and prints the table.
"""
import argparse
import csv
import os
import re
import sys

PATTERNS = {
    "total_formulas": re.compile(r'^Total Formulas:\s*(\d+)', re.M),
    "structure_rate": re.compile(r'^Structure Success Rate:\s*([\d.]+)%', re.M),
    "spacegroup_rate": re.compile(r'^Space Group Success Rate:\s*([\d.]+)%', re.M),
    "consensus_rate": re.compile(r'^Consensus Rate:\s*([\d.]+)%', re.M),
}


def parse_run(path):
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None

    out = {}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        if not m:
            return None
        out[key] = m.group(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--topn_dir", required=True,
                    help="Directory holding match_top<N>.out for each N.")
    ap.add_argument("--top_n_list", required=True,
                    help="Space-separated N values, e.g. '1 5 10 25'.")
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    ns = [int(x) for x in args.top_n_list.split()]

    rows = []
    missing = []
    for n in ns:
        parsed = parse_run(os.path.join(args.topn_dir, "match_top%d.out" % n))
        if parsed is None:
            missing.append(n)
            continue
        parsed["top_n"] = n
        parsed["model"] = args.tag
        rows.append(parsed)

    if not rows:
        sys.exit("No top-N results could be parsed under %s" % args.topn_dir)

    fields = ["model", "top_n", "total_formulas",
              "structure_rate", "spacegroup_rate", "consensus_rate"]
    with open(args.output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print()
    print("=" * 62)
    print("MATCH RATE vs NUMBER OF CANDIDATES -- %s" % args.tag)
    print("=" * 62)
    print("%6s  %10s  %12s  %11s" % ("top-N", "structure", "space group", "consensus"))
    print("-" * 62)
    for r in rows:
        print("%6d  %9s%%  %11s%%  %10s%%" % (
            r["top_n"], r["structure_rate"], r["spacegroup_rate"], r["consensus_rate"]))
    print("-" * 62)
    print("denominator: %s formulas" % rows[0]["total_formulas"])
    if missing:
        print("NOT PARSED : top-%s (no match_top<N>.out found)"
              % ", top-".join(str(m) for m in missing))
    print()
    print("saved: %s" % args.output_csv)


if __name__ == "__main__":
    main()
