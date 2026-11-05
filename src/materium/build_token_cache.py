#!/usr/bin/env python3
"""Build the token cache once, before the strategy runs start.

CrystalDataset writes its cache with a bare joblib.dump to the final path and
decides whether to rebuild from os.path.exists. Several training jobs starting
together would interleave writes into one file, and a job that checked midway
would load a half-written cache. Building it here, with the strategy runs
dependent on this job, makes that impossible.

The tokenizer must be constructed exactly as train.py constructs it, because the
cache key encodes those choices and any mismatch silently rebuilds.

Everything lives under a __main__ guard: tokenization uses a 'spawn' process
pool, and spawned children re-import this module. Without the guard they would
re-execute the build and die with "An attempt has been made to start a new
process before the current process has finished its bootstrapping phase".
"""
import os
import sys

from pymatgen.core import Element

from materium.datasets import CrystalDataset, MatterGenDataset
from materium.tokenizer import CrystalTokenizer, SequenceOrder, SortingOrder
from materium.wyckoff_tokenizer import WyckoffTokenizer

ELEMENT_VOCAB = [Element.from_Z(i).symbol for i in range(1, 100)]
LATTICE_STATS = {
    "a": (2.0, 10.0), "b": (2.0, 12.5), "c": (2.0, 20.0),
    "alpha": (60.0, 120.0), "beta": (60.0, 120.0), "gamma": (60.0, 120.0),
}


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mp_20_heldout"
    ox_mode = sys.argv[2] if len(sys.argv) > 2 else "guess"
    legacy_grid = len(sys.argv) > 3 and sys.argv[3] == "legacy"
    wyckoff = len(sys.argv) > 3 and sys.argv[3] in ("wyckoff", "wyckoff_canonical")
    canonical = len(sys.argv) > 3 and sys.argv[3] == "wyckoff_canonical"

    # train.py resolves its data directory relative to its own module, which
    # lives beside this one.
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if wyckoff:
        # Must match train.py --wyckoff exactly, cache prefix included.
        tokenizer = WyckoffTokenizer(
            element_vocab=ELEMENT_VOCAB,
            lattice_stats=LATTICE_STATS,
            num_quant_bins=1024,
            oxidation_mode=ox_mode,
            canonical=canonical,
        )
        prefix = "wyck2" if canonical else "wyck1"
    else:
        tokenizer = CrystalTokenizer(
            element_vocab=ELEMENT_VOCAB,
            lattice_stats=LATTICE_STATS,
            num_quant_bins=1024,
            sorting_order=SortingOrder.REVERSE_SPECIES,
            sequence_order=SequenceOrder.ATOMS_FIRST,
            oxidation_mode=ox_mode,
            legacy_bin_grid=legacy_grid,
        )
        prefix = "v3" if legacy_grid else "v4grid"
    print("dataset=%s ox=%s grid=%s" % (dataset, ox_mode, prefix), flush=True)

    # train.py reads the validation split but names its cache "test".
    for split, tag in (("train", "train"), ("val", "test")):
        ds = MatterGenDataset(os.path.join(data_root, dataset, split),
                              recalculate_cache=False, lattice_scaler=None)
        name = ("%s_oxy_crystal_token_cache_latticelast_%s_%s_%s_so%s_seq%s.joblib"
                % (prefix, dataset, ox_mode, tag,
                   tokenizer.sorting_order.value, tokenizer.sequence_order))
        cd = CrystalDataset(ds, tokenizer, cache_path=name, recalculate_cache=False)
        print("%s: %d sequences -> %s" % (split, len(cd), cd.cache_path), flush=True)

    print("CACHE READY")


if __name__ == "__main__":
    main()
