#!/usr/bin/env python3
"""Which CSPBench structures fail, under which models, and what they have in common.

Match labels are not recomputed: they are read from each run's official
match_results_<tag>_top<N>.csv, so every number here agrees with the summaries
already logged. What this adds is, per benchmark structure, a set of features
of the groundtruth, and per failed (model, formula) pair, a diagnosis of the
top-25 candidates the model did produce:

  groundtruth   atoms, elements, space group, crystal system, centering,
                Wyckoff orbits and free parameters, chemistry class, and how
                familiar the structure type is from mp_20_heldout/train
                (same space group; same anonymous formula; both = prototype).

  candidates    how many survived relaxation, whether any had the right space
                group, the modal space group, the share that is P1, whether a
                looser matcher (ltol 0.3, stol 0.5, angle 10) would accept one
                -- a near miss rather than a wrong structure -- and the median
                volume per atom relative to the groundtruth.

    python failure_analysis.py <runs_root> <groundtruth_dir> <heldout_train_dir> <out_dir> [workers]
"""
import collections
import csv
import functools
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import warnings

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from materium.datasets import get_spacegroup_int_number_fast
from materium.wyckoff import centering, encode, free_lattice_indices

warnings.filterwarnings("ignore")

# Manuscript labels -> evaluated run tags (Table: changes). All mp_20_heldout.
RUNS = [
    ("C1", "s1_ce_clean_ep159"),
    ("C2", "s2_soft_s2_ep311"),
    ("C3", "s3_soft_s64_ep251"),
    ("C4", "s7_ce_toss_ep342"),
    ("C5", "s6_soft_hardmix_ep107"),
    ("C6", "s4_ce_sgaux_ep116"),
    ("C7", "s5_soft_sgaux_ep359"),
]
SYMPREC = 0.1
LOOSE = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)


def sg_number(s):
    try:
        return SpacegroupAnalyzer(s, symprec=SYMPREC).get_space_group_number()
    except Exception:
        return 0


def crystal_system(number):
    # The seven systems; materium.wyckoff folds trigonal into hexagonal because
    # they share lattice parameters, but here the distinction is the point.
    for hi, name in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                     (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic")):
        if number <= hi:
            return name
    return ""


def anon(comp):
    return Composition(comp).reduced_composition.anonymized_formula


def bin_label(x, edges, lo=0):
    """'lo-hi' label of the bin x falls in; edges are inclusive upper bounds."""
    for hi in edges:
        if x <= hi:
            return "%d-%d" % (lo, hi) if lo != hi else str(hi)
        lo = hi + 1
    return ">%d" % edges[-1]


# ---------------------------------------------------------------- groundtruth
@functools.lru_cache(maxsize=None)
def load_gt(gt_dir, formula):
    s = Structure.from_file(os.path.join(gt_dir, formula + ".cif"))
    s.remove_oxidation_states()
    return s


def train_familiarity(train_dir):
    z = np.load(os.path.join(train_dir, "atomic_numbers.npy"))
    n = np.load(os.path.join(train_dir, "num_atoms.npy"))
    with open(os.path.join(train_dir, "space_group.json")) as fh:
        sgs = [get_spacegroup_int_number_fast(g) for g in json.load(fh)["values"]]
    off = np.concatenate([[0], np.cumsum(n)])
    by_sg, by_anon, by_proto = collections.Counter(), collections.Counter(), collections.Counter()
    for i in range(len(n)):
        c = collections.Counter(int(a) for a in z[off[i]:off[i + 1]])
        a = anon({Element.from_Z(k).symbol: v for k, v in c.items()})
        by_sg[sgs[i]] += 1
        by_anon[a] += 1
        by_proto[(a, sgs[i])] += 1
    return by_sg, by_anon, by_proto, len(n)


def gt_features(gt_dir, formula, fam):
    by_sg, by_anon, by_proto, n_train = fam
    s = load_gt(gt_dir, formula)
    sg = sg_number(s)
    comp = s.composition
    els = [e for e in comp.elements]
    f = {
        "formula": formula,
        "n_atoms": len(s),
        "n_elements": len(els),
        "sg": sg,
        "sg_symbol": SpacegroupAnalyzer(s, symprec=SYMPREC).get_space_group_symbol(),
        "crystal_system": crystal_system(sg),
        "centering": centering(sg),
        "vol_per_atom": round(s.volume / len(s), 3),
        "intermetallic": all(e.is_metal for e in els),
        "anon_formula": anon(comp),
        "train_same_sg": by_sg[sg],
        "train_same_anon": by_anon[anon(comp)],
        "train_prototype": by_proto[(anon(comp), sg)],
    }
    try:
        wc = encode(s, SYMPREC)
        f["n_orbits"] = len(wc.sites)
        f["n_free_coords"] = sum(len(x.free) for x in wc.sites)
        f["n_free_lattice"] = len(free_lattice_indices(wc.space_group))
    except Exception:
        f["n_orbits"] = f["n_free_coords"] = f["n_free_lattice"] = -1
    f["n_dof"] = f["n_free_coords"] + f["n_free_lattice"] if f["n_orbits"] >= 0 else -1
    try:
        f["charge_balanced"] = len(comp.oxi_state_guesses(max_sites=-50)) > 0
    except Exception:
        f["charge_balanced"] = False
    return f


# ----------------------------------------------------------------- candidates
def diagnose(job):
    runs_root, gt_dir, label, tag, formula, gt_sg = job
    gt = load_gt(gt_dir, formula)
    d = os.path.join(runs_root, tag, "topn", "top25", formula)
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    sgs, vols, loose = [], [], False
    for name in files:
        try:
            p = Structure.from_file(os.path.join(d, name))
            p.remove_oxidation_states()
        except Exception:
            continue
        sgs.append(sg_number(p))
        vols.append((p.volume / len(p)) / (gt.volume / len(gt)))
        if not loose:
            try:
                loose = LOOSE.fit(gt, p)
            except Exception:
                pass
    mode_sg, mode_n = collections.Counter(sgs).most_common(1)[0] if sgs else (0, 0)
    return {
        "model": label,
        "formula": formula,
        "n_candidates": len(sgs),
        "any_right_sg": gt_sg in sgs,
        "n_right_sg": sgs.count(gt_sg),
        "modal_sg": mode_sg,
        "modal_sg_share": round(mode_n / len(sgs), 2) if sgs else 0,
        "modal_crystal_system": crystal_system(mode_sg) if mode_sg else "",
        "right_crystal_system": crystal_system(gt_sg) in {crystal_system(x) for x in sgs if x},
        "p1_share": round(sgs.count(1) / len(sgs), 2) if sgs else 0,
        "loose_match": loose,
        "median_vol_ratio": round(statistics.median(vols), 3) if vols else float("nan"),
    }


# -------------------------------------------------------------------- summary
def read_labels(runs_root, tag, n):
    path = os.path.join(runs_root, tag, "match_results_%s_top%d.csv" % (tag, n))
    with open(path) as fh:
        return {r["formula"]: (r["struct_match"] == "True", r["sg_match"] == "True")
                for r in csv.DictReader(fh)}


def group_table(rows, key, labels, title, order=None):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    keys = order if order else sorted(groups, key=lambda k: (-len(groups[k]), str(k)))
    out = ["", "### %s" % title, "",
           "| group | formulas | mean structure % (C1-C7) | mean SG % | matched by none | "
           + " | ".join(labels) + " |",
           "|---|---|---|---|---|" + "---|" * len(labels)]
    for k in keys:
        g = groups.get(k)
        if not g:
            continue
        per = [100.0 * sum(r["str_" + m] for r in g) / len(g) for m in labels]
        sg = statistics.mean(100.0 * sum(r["sg_" + m] for r in g) / len(g) for m in labels)
        none = sum(1 for r in g if r["k_struct"] == 0)
        out.append("| %s | %d | %.1f | %.1f | %d (%.0f%%) | %s |" % (
            k, len(g), statistics.mean(per), sg, none, 100.0 * none / len(g),
            " | ".join("%.0f" % p for p in per)))
    return out


def main():
    runs_root, gt_dir, train_dir, out_dir = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.makedirs(out_dir, exist_ok=True)
    labels = [m for m, _ in RUNS]

    top25 = {m: read_labels(runs_root, t, 25) for m, t in RUNS}
    top1 = {m: read_labels(runs_root, t, 1) for m, t in RUNS}
    formulas = sorted(top25["C1"])
    assert len(formulas) == 180, len(formulas)

    fam = train_familiarity(train_dir)
    print("train structures %d, distinct prototypes %d" % (fam[3], len(fam[2])), flush=True)

    rows = []
    for f in formulas:
        r = gt_features(gt_dir, f, fam)
        for m in labels:
            r["str_" + m], r["sg_" + m] = top25[m][f]
            r["top1_str_" + m] = top1[m][f][0]
        r["k_struct"] = sum(r["str_" + m] for m in labels)
        r["k_sg"] = sum(r["sg_" + m] for m in labels)
        rows.append(r)
    print("features done", flush=True)

    jobs = [(runs_root, gt_dir, m, t, r["formula"], r["sg"])
            for m, t in RUNS for r in rows if not r["str_" + m]]
    with mp.Pool(workers) as pool:
        diags = pool.map(diagnose, jobs, chunksize=4)
    print("diagnosed %d failed (model, formula) pairs" % len(diags), flush=True)

    with open(os.path.join(out_dir, "per_formula.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["k_struct"], r["formula"])))
    with open(os.path.join(out_dir, "failed_pairs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diags[0]))
        w.writeheader()
        w.writerows(diags)

    # ---------------------------------------------------------- summary.md
    L = ["# CSPBench failure analysis (top-25, C1-C7, mp_20_heldout)", ""]
    L.append("Match labels from the official match_results CSVs. k = number of the 7 models "
             "that match the structure.")
    L += ["", "## How many models match each structure", "",
          "| k (models matching) | formulas (structure) | formulas (space group) |", "|---|---|---|"]
    for k in range(len(labels) + 1):
        L.append("| %d | %d | %d |" % (k, sum(r["k_struct"] == k for r in rows),
                                        sum(r["k_sg"] == k for r in rows)))

    L += ["", "## Pattern tables (structure match unless stated)"]
    L += group_table(rows, lambda r: r["crystal_system"], labels, "Crystal system",
                     ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
                      "trigonal", "hexagonal", "cubic"])
    na_order = ["1-4", "5-8", "9-12", "13-20", ">20"]
    L += group_table(rows, lambda r: bin_label(r["n_atoms"], [4, 8, 12, 20], lo=1), labels,
                     "Atoms in the primitive cell", na_order)
    L += group_table(rows, lambda r: r["n_elements"], labels, "Number of elements", [1, 2, 3, 4, 5, 6])
    L += group_table(rows, lambda r: bin_label(r["n_dof"], [0, 2, 5, 10]) if r["n_dof"] >= 0 else "n/a",
                     labels, "Free parameters (Wyckoff coords + lattice)",
                     ["0", "1-2", "3-5", "6-10", ">10", "n/a"])
    L += group_table(rows, lambda r: bin_label(r["n_orbits"], [1, 2, 4, 8], lo=1) if r["n_orbits"] >= 0 else "n/a",
                     labels, "Wyckoff orbits", ["1", "2", "3-4", "5-8", ">8", "n/a"])
    L += group_table(rows, lambda r: bin_label(r["train_prototype"], [0, 4, 19, 99]), labels,
                     "Prototype (anonymous formula + space group) count in mp_20_heldout/train",
                     ["0", "1-4", "5-19", "20-99", ">99"])
    L += group_table(rows, lambda r: bin_label(r["train_same_sg"], [49, 199, 999]), labels,
                     "Space-group count in mp_20_heldout/train",
                     ["0-49", "50-199", "200-999", ">999"])
    L += group_table(rows, lambda r: "intermetallic" if r["intermetallic"] else "has non-metal", labels,
                     "Chemistry class", ["intermetallic", "has non-metal"])
    L += group_table(rows, lambda r: "balanced" if r["charge_balanced"] else "no balanced assignment",
                     labels, "Charge-balanced oxidation assignment exists",
                     ["balanced", "no balanced assignment"])
    L += group_table(rows, lambda r: {1: "P", 2: "A/B/C/I", 3: "R", 4: "F"}[r["centering"]], labels,
                     "Centering", ["P", "A/B/C/I", "R", "F"])

    # failure modes
    L += ["", "## Failure modes over failed (model, formula) pairs", "",
          "| model | failed | no candidates | right SG among candidates | right crystal system | "
          "loose-match near miss | modal SG is P1 | median vol ratio |",
          "|---|---|---|---|---|---|---|---|"]
    for m in labels + ["all"]:
        ds = [d for d in diags if m == "all" or d["model"] == m]
        if not ds:
            continue
        n = len(ds)
        vr = [d["median_vol_ratio"] for d in ds if not math.isnan(d["median_vol_ratio"])]
        L.append("| %s | %d | %d | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.3f |" % (
            m, n, sum(d["n_candidates"] == 0 for d in ds),
            100.0 * sum(d["any_right_sg"] for d in ds) / n,
            100.0 * sum(d["right_crystal_system"] for d in ds) / n,
            100.0 * sum(d["loose_match"] for d in ds) / n,
            100.0 * sum(d["modal_sg"] == 1 for d in ds) / n,
            statistics.median(vr) if vr else float("nan")))

    # the lists
    by_f = collections.defaultdict(list)
    for d in diags:
        by_f[d["formula"]].append(d)
    cols = ("formula", "n_atoms", "n_elements", "sg_symbol", "crystal_system", "n_orbits",
            "n_dof", "train_prototype", "intermetallic")
    for title, sel in (("Matched by no model (k = 0)", lambda r: r["k_struct"] == 0),
                       ("Matched by 1-3 models (unstable)", lambda r: 1 <= r["k_struct"] <= 3)):
        L += ["", "## " + title, "",
              "| " + " | ".join(cols) + " | right SG seen (models) | near miss (models) | modal SGs | models matching |",
              "|" + "---|" * (len(cols) + 4)]
        for r in sorted((r for r in rows if sel(r)), key=lambda r: (r["crystal_system"], r["n_atoms"])):
            ds = by_f[r["formula"]]
            L.append("| " + " | ".join(str(r[c]) for c in cols) + " | %d | %d | %s | %s |" % (
                sum(d["any_right_sg"] for d in ds), sum(d["loose_match"] for d in ds),
                ",".join(str(x) for x, _ in collections.Counter(d["modal_sg"] for d in ds).most_common(3)),
                ",".join(m for m in labels if r["str_" + m]) or "-"))

    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
