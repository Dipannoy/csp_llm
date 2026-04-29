"""
One-time migration: replaces the plain nn.Embedding(230, 512) space_group
condition layer with a symmetry-aware SpaceGroupEmbedding (16-dim fingerprint
projected through a small MLP), warm-started to approximate the old weights.

Run from the repo root:
    python src/materium/migrate_sg_fingerprint.py

No GPU required — the warm-start optimises only 34 K parameters over 1000 steps.
Takes < 60 seconds on CPU.
"""

import os
import sys

# Ensure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from materium.model import LLamaTransformer

SRC = (
    "/work/dg47/MLEG/materium/materium/src/materium/checkpoints/"
    "alex_mp_toss_tokenizer_rev_species_atoms_first_material_llama_adamw_condition_latticelast_dim512/"
    "model_92_0.853.pt"
)

# Destination sits in the same checkpoint dir for easy reference
DST = (
    "/work/dg47/MLEG/materium/materium/src/materium/checkpoints/"
    "alex_mp_toss_tokenizer_rev_species_atoms_first_material_llama_adamw_condition_latticelast_dim512/"
    "model_92_0.853_sg_fingerprint.pt"
)

if __name__ == "__main__":
    LLamaTransformer.migrate_to_sg_fingerprint(
        src_ckpt_path=SRC,
        dst_ckpt_path=DST,
        condition_name="space_group",
        n_fit_steps=1000,   # more steps → better warm-start approximation
        fit_lr=1e-3,
    )
    print("\nDone. Migrated checkpoint:", DST)
    print(
        "Next step: run finetune_sg_fingerprint.sh to freeze all weights except the\n"
        "projector for 1 epoch, then optionally unfreeze for a second epoch at LR 1e-5."
    )

