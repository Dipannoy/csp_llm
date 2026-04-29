"""
Stage 2 & 3 fine-tuner for SpaceGroupEmbedding.

Stage 2 (projector-only, default):
    Freezes all weights except cond_proj_layers.space_group.projector (~34 K params).
    Use a higher LR (1e-3) since so few params are moving.

Stage 3 (full model, --freeze_projector_only=False):
    Unfreezes all weights. Use a lower LR (1e-5) to avoid destroying the
    pre-trained representations. Load from Stage 2's last_checkpoint.pt.

Run directly on an interactive node:
    # Stage 2
    python src/materium/finetune_sg_projector.py
    # Stage 3
    python src/materium/finetune_sg_projector.py \
        --freeze_projector_only=False --lr=1e-5 \
        --ckpt=<stage2_out_dir>/last_checkpoint.pt \
        --out_dir=<stage3_out_dir>

Or via finetune_sg_fingerprint.sh which runs all three stages sequentially.
"""

import argparse
import os
import sys
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from pymatgen.core import Element

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from materium.model import LLamaTransformer, SpaceGroupEmbedding
from materium.tokenizer import CrystalTokenizer, SortingOrder, SequenceOrder
from materium.datasets import MatterGenDataset, CrystalDataset, pad_collate_fn
from materium.train import run_epoch  # reuse the existing training loop

CKPT_DIR = os.path.join(
    "/work/dg47/MLEG/materium/materium/src/materium/checkpoints",
    "alex_mp_toss_tokenizer_rev_species_atoms_first_material_llama_adamw_condition_latticelast_dim512",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projector-only SG fine-tune")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(CKPT_DIR, "model_92_0.853_sg_fingerprint.pt"),
        help="Path to the migrated checkpoint",
    )
    parser.add_argument("--dataset", type=str, default="alex_mp_20_no_aug")
    parser.add_argument("--oxidation_mode", type=str, default="toss")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="LR for stage 2 (projector-only). Use 1e-5 for stage 3 (full model).",
    )
    parser.add_argument(
        "--freeze_projector_only",
        type=lambda x: x.lower() != "false",
        default=True,
        help="True = freeze all except SG projector (stage 2). False = train all (stage 3).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints. Defaults to a subfolder of CKPT_DIR.",
    )
    args = parser.parse_args()

    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load model ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    model, ckpt_meta = LLamaTransformer.load(args.ckpt, strict=True)

    sg_layer = model.cond_proj_layers.get("space_group")
    if not isinstance(sg_layer, SpaceGroupEmbedding):
        raise RuntimeError(
            f"Checkpoint does not contain a SpaceGroupEmbedding for 'space_group'. "
            f"Got: {type(sg_layer)}. Run migrate_sg_fingerprint.py first."
        )

    # ── 2. Freeze / unfreeze ────────────────────────────────────────────────
    projector_params = set(id(p) for p in sg_layer.projector.parameters())
    frozen_count = trainable_count = 0

    if args.freeze_projector_only:
        # Stage 2: only the projector is trainable
        for name, param in model.named_parameters():
            if id(param) in projector_params:
                param.requires_grad_(True)
                trainable_count += param.numel()
            else:
                param.requires_grad_(False)
                frozen_count += param.numel()
        stage_label = "stage2_projector_only"
        print(
            f"Stage 2 — projector-only fine-tune\n"
            f"  Frozen:    {frozen_count:,} parameters\n"
            f"  Trainable: {trainable_count:,} parameters (SG projector)"
        )
    else:
        # Stage 3: all weights trainable
        for param in model.parameters():
            param.requires_grad_(True)
            trainable_count += param.numel()
        stage_label = "stage3_full_model"
        print(
            f"Stage 3 — full model fine-tune\n"
            f"  Trainable: {trainable_count:,} parameters (all weights)"
        )

    model = model.to(DEVICE)

    # ── 3. Build tokenizer (same config as original training) ──────────────
    ELEMENT_VOCAB = [Element.from_Z(i).symbol for i in range(1, 100)]
    LATTICE_STATS = {
        "a": (2.0, 10.0),
        "b": (2.0, 12.5),
        "c": (2.0, 20.0),
        "alpha": (60.0, 120.0),
        "beta": (60.0, 120.0),
        "gamma": (60.0, 120.0),
    }

    tokenizer = CrystalTokenizer(
        element_vocab=ELEMENT_VOCAB,
        lattice_stats=LATTICE_STATS,
        num_quant_bins=1024,
        sorting_order=SortingOrder.REVERSE_SPECIES,
        sequence_order=SequenceOrder.ATOMS_FIRST,
        oxidation_mode=args.oxidation_mode,
    )

    # ── 4. Build datasets (reuse cached tokens — no re-tokenisation) ────────
    cwd = os.path.join(os.path.dirname(__file__), "data")
    train_dataset = MatterGenDataset(
        os.path.join(cwd, args.dataset, "train"), recalculate_cache=False
    )
    val_dataset = MatterGenDataset(
        os.path.join(cwd, args.dataset, "val"), recalculate_cache=False
    )

    cache_suffix = (
        f"v3_oxy_crystal_token_cache_latticelast_{args.dataset}_{args.oxidation_mode}"
        f"_so{tokenizer.sorting_order.value}_seq{tokenizer.sequence_order.value}"
    )
    train_crystals = CrystalDataset(
        train_dataset,
        tokenizer,
        cache_path=f"{cache_suffix}_train.joblib",
        recalculate_cache=False,
    )
    val_crystals = CrystalDataset(
        val_dataset,
        tokenizer,
        cache_path=f"{cache_suffix}_test.joblib",
        recalculate_cache=False,
    )

    pad_id = tokenizer._special_to_id["[PAD]"]
    train_loader = DataLoader(
        train_crystals,
        batch_size=args.batch,
        collate_fn=lambda b: pad_collate_fn(b, pad_token_id=pad_id),
        num_workers=2,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_crystals,
        batch_size=args.batch,
        collate_fn=lambda b: pad_collate_fn(b, pad_token_id=pad_id),
        num_workers=2,
        shuffle=False,
    )

    # ── 5. Optimizer ────────────────────────────────────────────────────────
    if args.freeze_projector_only:
        # Only optimise the projector
        opt_params = sg_layer.projector.parameters()
    else:
        # Optimise everything; give the SG projector a slightly higher LR
        opt_params = [
            {"params": sg_layer.projector.parameters(), "lr": args.lr * 3},
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "space_group" not in n
                ],
                "lr": args.lr,
            },
        ]

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    # ── 6. W&B logging ──────────────────────────────────────────────────────
    wandb.login(
        key="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",
        relogin=True,
    )
    wandb.init(
        entity="dipannoydip",
        project="materium_bertos",
        name=f"sg_fingerprint_{stage_label}",
        config={
            "ckpt": args.ckpt,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "freeze_projector_only": args.freeze_projector_only,
            "trainable_params": trainable_count,
        },
    )

    # ── 7. Training loop ─────────────────────────────────────────────────────
    default_out_dir = os.path.join(CKPT_DIR, f"sg_fingerprint_{stage_label}")
    out_dir = args.out_dir if args.out_dir else default_out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving checkpoints to: {out_dir}")

    best_val_loss = float("inf")
    start_epoch = ckpt_meta.get("epoch", -1) + 1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = run_epoch(model, optimizer, train_loader, DEVICE, is_train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, optimizer, val_loader, DEVICE, is_train=False)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(
                out_dir, f"sg_fp_{stage_label}_{epoch}_{val_loss:.3f}.pt"
            )
            model.save(
                save_path,
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                loss=val_loss,
                tokenizer=tokenizer.to_dict(),
            )
            print(f"  Saved best model \u2192 {save_path}")

        # Always save last checkpoint so the next stage can pick it up
        last_path = os.path.join(out_dir, "last_checkpoint.pt")
        model.save(
            last_path,
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            loss=val_loss,
            tokenizer=tokenizer.to_dict(),
        )

    print(f"Fine-tune complete. Last checkpoint: {last_path}")

