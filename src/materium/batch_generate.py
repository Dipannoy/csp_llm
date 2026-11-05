#!/usr/bin/env python3
"""Generate candidate structures for many targets in one process.

llm_generate.py loads the model, builds trackers and writes bar charts per
invocation. That is fine for 180 formulas and unusable for 9,046: it would
reload the checkpoint 9,046 times and spend most of its life in matplotlib.
This loads the model once, drives generate_structure directly, and writes CIFs.

Throughput comes from sharding: targets[shard::num_shards] means N array tasks
split the work with no coordination, and each one holds the model for its whole
share.

Input is a TSV of `key<TAB>formula` (further columns ignored), so it serves both
benchmarks: CSPBench keys by formula, the MP-20 split keys by material_id
because its 9,046 structures repeat compositions.

Output: <out_dir>/<key>/<key>_<i>.cif
"""
import argparse
import os
import sys
import time

import torch

from materium.generation_utils import generate_structure
from materium.model import LLamaTransformer
from materium.tokenizer import tokenizer_from_dict
from materium.wyckoff_generation import generate_wyckoff_structure


def load(ckpt_dir, ckpt_file, device):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    path = os.path.join(root, ckpt_dir, ckpt_file)
    if not os.path.isfile(path):
        sys.exit("checkpoint not found: %s" % path)

    model, ckpt = LLamaTransformer.load(path, strict=False)
    cfg = ckpt.get("tokenizer")
    if cfg is None:
        sys.exit("checkpoint carries no tokenizer config: %s" % path)
    tok = tokenizer_from_dict(cfg)
    model = model.to(device).eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--ckpt_file", required=True)
    ap.add_argument("--targets", required=True, help="TSV: key<TAB>formula")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_weight", type=float, default=2.5)
    ap.add_argument("--csp_oxi_mode", default=None,
                    help="bertos applies the BERTOS per-element species constraint")
    ap.add_argument("--formula_mode", choices=["strict", "free"], default="strict")
    ap.add_argument("--resume", action="store_true",
                    help="Skip targets that already have num_samples CIFs.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = []
    with open(args.targets) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                targets.append((parts[0], parts[1]))
    mine = targets[args.shard::args.num_shards]

    print("targets %d, shard %d of %d -> %d, samples %d, device %s"
          % (len(targets), args.shard, args.num_shards, len(mine),
             args.num_samples, device), flush=True)

    model, tok = load(args.ckpt_dir, args.ckpt_file, device)
    constrain = args.formula_mode == "strict"
    gen = generate_wyckoff_structure if getattr(tok, "is_wyckoff", False) else generate_structure

    t0 = time.time()
    done = written = invalid = skipped = 0

    # inference_mode is slightly stronger than no_grad and matters here because
    # the sampling loop runs once per token, tens of millions of times overall.
    with torch.inference_mode():
        for key, formula in mine:
            out = os.path.join(args.out_dir, key)
            if args.resume and len(
                [f for f in os.listdir(out) if f.endswith(".cif")]
                if os.path.isdir(out) else []
            ) >= args.num_samples:
                skipped += 1
                continue
            os.makedirs(out, exist_ok=True)

            for i in range(args.num_samples):
                try:
                    toks = gen(
                        model, tok,
                        device=device,
                        temperature=args.temperature,
                        conditions={"reduced_formula": formula},
                        classifier_guidance_weight=args.cfg_weight,
                        csp_oxi_mode=args.csp_oxi_mode,
                        constrain_formula=constrain,
                    )
                    struct = tok.detokenize(toks)
                    struct.to(filename=os.path.join(out, "%s_%d.cif" % (key, i)),
                              fmt="cif")
                    written += 1
                except Exception:
                    # A rejected sample is normal: the sequence may be truncated
                    # or carry a token the detokenizer refuses. Counted, not
                    # printed, because at this scale printing dominates runtime.
                    invalid += 1

            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print("  %d/%d targets  %d cifs  %d invalid  %.1f s/target"
                      % (done, len(mine), written, invalid, el / done), flush=True)

    el = time.time() - t0
    print()
    print("shard %d done: %d targets (%d skipped), %d cifs, %d invalid, %.1f min"
          % (args.shard, done, skipped, written, invalid, el / 60.0))
    if done:
        print("rate: %.2f s/target, %.3f s/sample" % (el / done, el / max(written, 1)))


if __name__ == "__main__":
    main()
