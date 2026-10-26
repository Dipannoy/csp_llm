#!/usr/bin/env python3
"""Launch a CSPBench evaluation for each strategy as soon as it has a usable
checkpoint, so the five loss candidates can be compared without waiting for any
of them to reach 500 epochs.

Checkpoints are written as model_<epoch>_<valloss>.pt only when validation loss
improves, so the highest-epoch file is also the best one so far. This waits for
a checkpoint at or past --min_epoch, takes the lowest validation loss among
those, and submits the generate/relax/match pipeline on it.

Idempotent: every submission is recorded, so running this repeatedly never
launches the same evaluation twice. That is what lets it run on a timer.
"""
import argparse
import os
import re
import subprocess
import sys

CKPT_RE = re.compile(r"^model_(\d+)_([\d.]+)\.pt$")

# train.py builds the directory name from the tokenizer's sort and sequence
# order plus the hidden size, not from CHECK_DIR alone.
DIR_SUFFIX = "_rev_species_atoms_first_material_llama_adamw_condition_latticelast_dim512"


def checkpoints(root, strategy):
    d = os.path.join(root, "src/materium/checkpoints", strategy + DIR_SUFFIX)
    if not os.path.isdir(d):
        return d, []
    found = []
    for name in os.listdir(d):
        m = CKPT_RE.match(name)
        if m:
            found.append((int(m.group(1)), float(m.group(2)), name))
    return d, found


def load_state(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/dg47/MLEG/materium/materium")
    ap.add_argument("--strategies", required=True, help="comma-separated CHECK_DIRs")
    ap.add_argument("--min_epoch", type=int, default=100)
    ap.add_argument("--state_file", default="benchmarks/auto_eval_state.txt")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--concurrent", type=int, default=6,
                    help="Array concurrency per evaluation. Kept low so several "
                         "evaluations and the training runs can coexist.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    state_path = os.path.join(args.root, args.state_file)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    done = load_state(state_path)

    launched = pending = 0
    for strategy in args.strategies.replace(":", ",").split(","):
        strategy = strategy.strip()
        if not strategy:
            continue

        ckpt_dir_full, found = checkpoints(args.root, strategy)
        ready = [c for c in found if c[0] >= args.min_epoch]

        if not ready:
            best = max(found)[0] if found else None
            print("%-22s waiting (best epoch %s, need %d)"
                  % (strategy, best if best is not None else "-", args.min_epoch))
            pending += 1
            continue

        # Lowest validation loss among the eligible checkpoints.
        epoch, loss, fname = min(ready, key=lambda c: c[1])
        key = "%s %s" % (strategy, fname)
        if key in done:
            print("%-22s already evaluated on %s" % (strategy, fname))
            continue
        # One evaluation per strategy, not one per checkpoint improvement. The
        # point is to rank the candidates early, and a later checkpoint of an
        # already-ranked strategy does not change that ranking enough to justify
        # another 9000 generations plus relaxations.
        if any(line.split(" ", 1)[0] == strategy for line in done):
            prev = [l for l in done if l.split(" ", 1)[0] == strategy][0]
            print("%-22s skipping %s; already evaluated (%s). Re-evaluate later "
                  "by removing its line from the state file."
                  % (strategy, fname, prev.split(" ", 1)[1]))
            continue

        tag = "%s_ep%d" % (strategy, epoch)
        env = dict(os.environ)
        env.update({
            "TAG": tag,
            # submit.sh passes CKPT_DIR straight to llm_generate.py, which
            # resolves it under src/materium/checkpoints -- so it needs the
            # suffixed directory name, not the bare strategy name.
            "CKPT_DIR": strategy + DIR_SUFFIX + "/",
            "CKPT_FILE": fname,
            "OXI_MODE": "bertos" if "toss" in strategy else "guess",
            "NUM_SAMPLES": str(args.num_samples),
            "CONCURRENT": str(args.concurrent),
            "FORMULA_MODE": "strict",
        })

        print("%-22s submitting %s (epoch %d, val %.3f) as TAG=%s"
              % (strategy, fname, epoch, loss, tag))
        if args.dry_run:
            continue

        r = subprocess.run(["./submit.sh"], cwd=os.path.join(args.root, "csp_pipeline"),
                           env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            print("%-22s SUBMIT FAILED rc=%d -- not recorded, will retry"
                  % (strategy, r.returncode))
            continue

        with open(state_path, "a") as fh:
            fh.write(key + "\n")
        launched += 1

    print()
    print("launched %d, still waiting %d" % (launched, pending))
    # Non-zero while any strategy has yet to produce an eligible checkpoint, so
    # the watcher loop knows there is still work to come.
    return 1 if pending else 0


if __name__ == "__main__":
    sys.exit(main())
