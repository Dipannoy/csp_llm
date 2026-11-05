#!/usr/bin/env python3
"""Ground-truth space groups for the CSPBench structures, for the oracle probe.

The model carries a space-group conditioning embedding that was trained (with
50% dropout) but that CSP inference never uses, because at inference the space
group is unknown. Conditioning on the TRUE space group measures the ceiling of
symmetry information: how much of the match-rate deficit disappears when the
model is told the one thing it is worst at.

Assigned with the same symprec the match-rate script uses, so "the oracle space
group" is the same label the space-group match is later judged against.

Writes `<formula>\t<sg number>\t<symbol>` keyed by the groundtruth file stem,
which is exactly the formula string the pipeline's formula list carries.

Also reports how many benchmark groups never occur in the training labels: the
conditioning embedding row for such a group was never trained, so the oracle can
carry no information for those formulas.
"""
import collections
import json
import os
import sys

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SYMPREC = 0.1   # matches calculate_success_rate.py


def main(groundtruth_dir, out_tsv, train_sg_json=None):
    rows = []
    for f in sorted(os.listdir(groundtruth_dir)):
        if not f.endswith(".cif"):
            continue
        s = Structure.from_file(os.path.join(groundtruth_dir, f))
        sga = SpacegroupAnalyzer(s, symprec=SYMPREC)
        rows.append((f[:-4], sga.get_space_group_number(), sga.get_space_group_symbol()))

    with open(out_tsv, "w") as fh:
        for r in rows:
            fh.write("%s\t%d\t%s\n" % r)
    print("wrote %d rows to %s" % (len(rows), out_tsv))

    nums = collections.Counter(r[1] for r in rows)
    print("distinct groups: %d" % len(nums))
    print("most common: %s" % nums.most_common(8))

    # Embedding(230) is indexed by the raw international number, so group 230
    # has no row at all and would crash generation outright.
    bad = [r for r in rows if not 1 <= r[1] <= 229]
    print("outside embedding range 1..229: %d %s" % (len(bad), bad))

    if train_sg_json:
        with open(train_sg_json) as fh:
            train_syms = set(json.load(fh)["values"])
        # Compare by symbol: the training file stores symbols, which the dataset
        # converts to numbers at load time.
        unseen = [r for r in rows if r[2] not in train_syms]
        print("benchmark formulas whose group never occurs in training: %d" % len(unseen))
        for r in unseen:
            print("  %s  %d  %s" % r)


if __name__ == "__main__":
    main(*sys.argv[1:4])
