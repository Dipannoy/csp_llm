# CSP pipeline: generate → relax → match

One submit command runs all three stages. Previously each ran from a different
directory in a different environment, and a formula lost between stages showed up
only as a missing row in the final CSV.

## Use

```sh
cd csp_pipeline
./submit.sh                 # submit or resume the run named in config.sh
./submit.sh --status        # per-formula progress, no submission
./submit.sh --only-aggregate  # recompute matching on whatever exists
```

Override anything from the environment:

```sh
TAG=mp20_baseline_strict FORMULA_MODE=strict \
CKPT_DIR=train_materium_mp20_guess_.../ CKPT_FILE=model_222_1.907.pt \
./submit.sh
```

## The three trees

| Tree | Path | Stage |
|---|---|---|
| `MATERIUM_ROOT` | `/work/dg47/MLEG/materium/materium` | generation (`llm_env` venv) |
| `MATRIS_ROOT` | `/work/dg47/MLEG/MatRIS` | relaxation (`matris_env` conda) |
| `METRICS_ROOT` | `/work/dg47/MLEG/CSPBenchMetrics` | matching (`metrics_2`, forced paths) |

Matching uses `calculate_success_rate.py` (the one that reports Consensus Rate),
not `csp_materium/check_matching_rate.py`.

Generation writes to `src/materium/qe_input/$TAG/` and relaxation reads straight
from it — the old copy into a `materium_structures` directory is gone.

## Structure

| File | Role |
|---|---|
| `config.sh` | every path, env and hyperparameter; the only file to edit |
| `submit.sh` | builds the formula list, submits the array + dependent aggregate |
| `run_formula.sh` | array task: generate then relax **one** formula |
| `run_aggregate.sh` | manifest, then match rate at each top-N |
| `pipeline_status.py` | reconciles the run against the 180-formula benchmark |
| `make_topn_view.py` | symlink view of the N lowest-energy structures |
| `summarize_topn.py` | collects the top-N rates into one table |

Artefacts land under `$MATERIUM_ROOT/csp_pipeline_runs/$TAG/`:

```
state/<FORMULA>/{generate,relax}.{ok,fail}   stage markers
relaxed/<FORMULA>/<FORMULA>_<rank>.cif       relaxed structures, rank 1 = lowest Ef
topn/top<N>/<FORMULA>/                       symlinks to ranks 1..N
topn/match_top<N>.out                        that budget's match output
manifest_<TAG>.csv                           per-formula outcome
match_results_<TAG>_top<N>.csv               per-formula match detail
topn_summary_<TAG>.csv                       the top-N table
formulas.tsv                                 array index == line number
logs/                                        per-task logs
```

## Match rate at top-N

`TOP_N_LIST="1 5 10 25"` in `config.sh`. For each N, a view directory of symlinks
to ranks 1..N is built and scored, so the unmodified `calculate_success_rate.py`
evaluates exactly the N most stable candidates:

```
 top-N   structure   space group    consensus
     1      31.11%        28.33%       26.11%
     5      48.89%        42.22%       40.00%
    10      53.33%        47.22%       45.00%
    25      57.22%        50.56%       49.44%
```

`TOP_PICK` defaults to `1.0`, not the `0.5` used before. It is the fraction of
relaxed structures saved, so `0.5` of 50 samples leaves exactly 25, and any
formula yielding fewer than 50 valid candidates silently caps below top-25.

A formula with fewer than N structures is scored on all it has, and
`make_topn_view.py` reports how many such formulas there are — so a top-25 rate
built mostly from formulas holding 10 structures is not mistaken for a true
top-25.

Ranks are sorted numerically, not lexicographically: a name sort would place
rank 10 before rank 2 and build the wrong top-5.

`relaxed/` deliberately sits outside the generation directory. When relaxation
output lived inside its own input, the folder walk re-ingested previous results
and reported 246 folders for a 180-formula benchmark.

## Nothing is ever repeated

Each stage writes its marker only after its output has been counted, so a task
killed mid-stage leaves no marker and is redone. A stage is considered complete
only if the marker exists **and** the CIFs are still on disk — a stale marker
over a deleted output does not cause a skip.

Resubmitting with the same `TAG` is the supported recovery path. Finished
formulas exit in under a second.

## One formula per array task

A single collapsed cell once made MatRIS try to allocate 17.98 GiB. The task
died, every formula its shard had not yet reached was lost, and the job still
reported `COMPLETED 0:0`. One formula per task bounds that to one formula, and
the marker records which stage it died in.

## Failures are counted, not dropped

`calculate_success_rate.py` builds `total_formulas` from
`os.scandir(pred_base_dir)`, so a formula with no output disappears from the
totals — that is how 180 formulas reported as 175, and a quoted 58.86% was really
57.22%. `pipeline_status.py --fill_empty` creates an empty directory for every
such formula before matching runs, so it scores as the failure it is.

`manifest_<TAG>.csv` gives the reason per formula:

| status | meaning |
|---|---|
| `ok` | relaxed structures exist |
| `no_valid_samples` | generation exited 0 but wrote no CIFs |
| `all_candidates_rejected` | every CIF failed composition or was unphysical |
| `failed` | non-zero exit; see that formula's log |
| `not_started` | stage never ran (task killed, or array incomplete) |

Find a formula's log through its line number in `formulas.tsv` — that is its
array index:

```sh
IDX=$(grep -n -m1 "^Mn9Au3" formulas.tsv | cut -d: -f1)
cat logs/formula_*_${IDX}.out
```
