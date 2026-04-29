import random
import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from typing import Dict
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from dataclasses import dataclass
from pymatgen.core import Lattice, Structure, Element, Composition
import os
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from muon import SingleDeviceMuonWithAuxAdam
from enum import Enum
from pymatgen.io.ase import AseAtomsAdaptor
from mattersim.forcefield.potential import MatterSimCalculator

from mattersim.applications.relax import Relaxer
from materium.generation_utils import generate_structure
from materium.model import ConditionConfig, LLamaTransformer, ModelArgs, RopeMode
from materium.datasets import (
    StandardScaler,
    get_spacegroup_int_number_fast,
    MatterGenDataset,
    CrystalDataset,
    pad_collate_fn,
)
from mattergen.evaluation.utils.utils import compute_rmsd_angstrom
from collections import Counter, deque
from mattergen.evaluation.evaluate import evaluate

from pymatgen.io.ase import AseAtomsAdaptor
from mattersim.forcefield.potential import MatterSimCalculator

from mattersim.applications.relax import Relaxer
import copy

from torch.utils.data import DataLoader


from materium.tokenizer import (
    CrystalTokenizer,
    SortingOrder,
    SequenceOrder,
    calculate_stats_from_dataset,
)


def relax_structure_mattersim(
    structure: Structure,
    fmax: float = 0.1,
    optimizer: str = "FIRE",
    filter_type: str = "ExpCellFilter", 
    constrain_symmetry: bool = True,
    max_steps: int = 500,
    device="cpu",
):

    ase_atoms = AseAtomsAdaptor.get_atoms(structure)
    ase_atoms.calc = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth", device=device
    )  
    relaxer = Relaxer(
        optimizer=optimizer,
        filter=filter_type,
        constrain_symmetry=constrain_symmetry,
    )

    relaxed_structure = relaxer.relax(ase_atoms, steps=max_steps, fmax=fmax)[1]
    pymatgen_structure = AseAtomsAdaptor.get_structure(relaxed_structure)
    total_energy = relaxed_structure.get_total_energy()
    forces = relaxed_structure.get_forces()
    energy_per_atom = total_energy / len(relaxed_structure.positions)
    return pymatgen_structure, total_energy, energy_per_atom, forces


def put_dict_on_device(d: Dict[str, torch.Tensor], device):

    return {
        k: (
            v.to(device)
            if isinstance(v, torch.Tensor)
            else put_dict_on_device(v, device)
        )
        for k, v in d.items()
    }


def build_position_weights(
    targets: torch.Tensor,
    lattice_tok_id: int,
    quant_offset: int,
    num_special_tokens: int,
    lattice_weight: float = 5.0,
    species_weight: float = 2.0,
    coord_weight: float = 1.0,
) -> torch.Tensor:
    """
    Returns a per-position weight tensor (same shape as `targets`) that
    upweights lattice tokens and element/species tokens relative to coordinate
    tokens.  Special tokens ([SOS], [ATOMS], [LATTICE], [EOS], [PAD]) get
    weight 1.0 (no boost) since they carry less structural information.

    Token type detection (from target IDs):
      - lattice quant : quant token that follows a [LATTICE] token  → lattice_weight
      - species       : special_count <= id < quant_offset          → species_weight
      - coord quant   : quant token before [LATTICE]                → coord_weight
      - special / pad : everything else                             → 1.0
    """
    B, S = targets.shape
    weights = torch.ones(B, S, dtype=torch.float32, device=targets.device)

    # Detect positions strictly after the first [LATTICE] token in each row.
    # cumsum trick: once [LATTICE] appears, all subsequent positions are > 0.
    lattice_marker = (targets == lattice_tok_id).long().cumsum(dim=1)  # [B, S]
    # Shift right by 1 so the [LATTICE] token itself is NOT considered inside the section
    in_lattice = torch.cat(
        [torch.zeros(B, 1, dtype=torch.long, device=targets.device), lattice_marker[:, :-1]],
        dim=1,
    ) > 0  # [B, S], bool

    is_quant = targets >= quant_offset                              # coord or lattice bin
    is_lattice_quant = is_quant & in_lattice
    is_coord_quant   = is_quant & ~in_lattice
    is_species = (targets >= num_special_tokens) & (targets < quant_offset)

    weights[is_lattice_quant] = lattice_weight
    weights[is_coord_quant]   = coord_weight
    weights[is_species]       = species_weight

    return weights


def run_epoch(
    model, optimizer, dataloader, device, is_train=True, ema_model=None, ema_decay=0.999,
    tokenizer=None, lattice_weight=5.0, species_weight=2.0, coord_weight=1.0,keep_conditions=None,
):
    """
    tokenizer: if provided, per-position loss weights are applied
               (lattice tokens ×lattice_weight, species ×species_weight,
                coord bins ×coord_weight).  Pass None to use flat cross-entropy.
    """
    all_loss = []
    if is_train:
        model.train()
    else:
        model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    desc = "Training" if is_train else "Validation"
    for batch_idx, batch in tqdm(
        enumerate(dataloader), desc=desc, total=len(dataloader)
    ):

        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):

            input_seq = batch["tokens"].to(device)
            target_seq = batch["targets"].to(device)
            conditions = batch["conditions"]
            condition_keys = list(
                [k for k in conditions.keys() if not k.endswith("_mask")]
            )
            for k in condition_keys:
                if k in (keep_conditions or set()):
                    continue  # never drop conditions that are the sole trainable path
                if np.random.random() < 0.5:
                    del conditions[k]
                    if f"{k}_mask" in condition_keys:
                        print("Also deleting", f"{k}_mask")
                        del conditions[f"{k}_mask"]

            conditions = put_dict_on_device(conditions, device)

            # Build per-position loss weights when tokenizer is supplied
            pos_weights = None
            if tokenizer is not None:
                pos_weights = build_position_weights(
                    target_seq,
                    lattice_tok_id=tokenizer._special_to_id["[LATTICE]"],
                    quant_offset=tokenizer.quant_offset,
                    num_special_tokens=len(tokenizer.special_tokens),
                    lattice_weight=lattice_weight,
                    species_weight=species_weight,
                    coord_weight=coord_weight,
                )

            logits = model(input_seq, targets=target_seq, conditions=conditions,
                           position_weights=pos_weights)
            loss = model.last_loss

        if is_train:
            # optimizer.zero_grad()
            
            if not loss.requires_grad:
                # All trainable parameters were disconnected from this batch
                # (e.g. the only trainable condition was randomly dropped).
                # Skip backward to avoid RuntimeError.
                all_loss.append(loss.item())
                continue
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        if p_ema.dtype == p.dtype:
                            p_ema.mul_(ema_decay).add_(
                                p.to(p_ema.device), alpha=1.0 - ema_decay
                            )
                            
        # if is_train:
        #     wandb.log({"train_loss": loss.item()})
        # else:
        #     wandb.log({"val_loss": loss.item()})

        all_loss.append(loss.item())

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Batch Idx {batch_idx} Loss (last 100): {np.mean(all_loss[-100:]):.4f}"
            )

    return np.mean(all_loss)


def init_ema(model, decay=0.999, device="cuda"):
    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema, decay


def train_model(
    model: LLamaTransformer,
    tokenizer: CrystalTokenizer,
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs,
    device,
    start_epoch=0,
    ckpt_path="ckpt",
):
    """The main training loop."""
    model.train()

    ema_model, ema_decay = None, 0.999
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
    print("--- Starting Training ---")
    best_val_loss = 1e10
    for epoch in range(start_epoch, epochs):
        mean_loss_train = run_epoch(
            model,
            optimizer,
            dataloader_train,
            device,
            is_train=True,
            ema_model=ema_model,
            ema_decay=ema_decay,
            tokenizer=tokenizer,
        )
        with torch.no_grad():
            mean_loss_test = run_epoch(
                model, optimizer, dataloader_test, device, is_train=False,
                tokenizer=tokenizer,
            )

        print(
            f"Epoch {epoch}/{epochs} Mean Train loss {mean_loss_train:.3f} Mean Test loss: {mean_loss_test:.3f}"
        )
        scheduler.step(mean_loss_test)
        
        wandb.log({
                "epoch": epoch,
                "train_loss": mean_loss_train,
                "val_loss": mean_loss_test
            })

        if mean_loss_test < best_val_loss:
            os.makedirs(ckpt_path, exist_ok=True)
            best_val_loss = mean_loss_test
            model_path = os.path.join(
                ckpt_path, f"model_{epoch}_{mean_loss_test:.3f}.pt"
            )
            model.save(
                model_path,
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                loss=mean_loss_test,
                tokenizer=tokenizer.to_dict(),
            )

            if ema_model is not None:

                with torch.no_grad():
                    mean_loss_test_ema = run_epoch(
                        ema_model.to(device),
                        optimizer,
                        dataloader_test,
                        device,
                        is_train=False,
                    )
                ema_model_path = os.path.join(
                    ckpt_path, f"ema_{epoch}_{mean_loss_test_ema:.3f}.pt"
                )
                print("Ema loss", mean_loss_test_ema)
                ema_model.save(
                    ema_model_path,
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    loss=mean_loss_test_ema,
                    tokenizer=tokenizer.to_dict(),
                )

            print(f"Saving new best model {model_path}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

            print("Generating samples")

            with torch.no_grad():
                for gen_idx in range(8):
                    try:
                        generated_material = generate_structure(
                            model,
                            tokenizer,
                            device=DEVICE,
                            conditions={},
                            temperature=0.5,
                        )
                        generated_material: Structure = tokenizer.detokenize(
                            generated_material
                        )
                        save_struct_path = os.path.join(ckpt_path, "generated")
                        os.makedirs(save_struct_path, exist_ok=True)
                        generated_material.to(
                            os.path.join(
                                save_struct_path, f"material_sample_{gen_idx}.cif"
                            ),
                            "cif",
                        )
                        print(generated_material)
                    except Exception as e:
                        print("Exception while generation", e)
        last_path = os.path.join(ckpt_path, "last_checkpoint.pt")
        model.save(
            last_path,
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            loss=mean_loss_test,
            tokenizer=tokenizer.to_dict(),
        )

    print("--- Training Complete ---")


def configure_optimizer(model: LLamaTransformer, type: str = "adamw"):
    if type.lower() == "muon":
        transformer_body = model.layers
        hidden_weights = [p for p in transformer_body.parameters() if p.ndim >= 2]

        other_params = [
            p for p in model.parameters() if all(p is not hw for hw in hidden_weights)
        ]

        param_groups = [
            # Group for hidden weights, to be optimized with Muon
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=0.01,  # Muon typically uses a higher LR
                weight_decay=0.1,
            ),
            # Group for all other parameters, to be optimized with AdamW
            dict(
                params=other_params,
                use_muon=False,
                lr=4e-4,  # Standard AdamW learning rate
                betas=(0.9, 0.95),
                weight_decay=0.2,
            ),
        ]
        # NOTE: atleast in this config the optimizer seems to be worse than the adamw. Maybe with more hyperparameter tuning it is better?
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=8e-5,
            betas=(0.9, 0.95),
            weight_decay=0.05,
            fused=torch.cuda.is_available(),
        )

    return optimizer
    
def load_model(args):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", args.check_dir)
    model, ckpt = LLamaTransformer.load(
        os.path.join(ckpt_path, args.ckpt_file), strict=False
    )

    # ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "llm_v2_material_llama_adamw_condition_latticelast_dim384")
    # model,ckpt = LLamaTransformer.load(os.path.join(ckpt_path,"model_41_0.815.pt"),strict=False)
    ELEMENT_VOCAB = [Element.from_Z(i).symbol for i in range(1, 100)]
    LATTICE_STATS = {
        "a": (2.0, 10.0),
        "b": (2.0, 12.5),
        "c": (2.0, 20.0),
        "alpha": (60.0, 120.0),
        "beta": (60.0, 120.0),
        "gamma": (60.0, 120.0),
    }
    if ckpt.get("tokenizer") is None:
        tokenizer = CrystalTokenizer(
            element_vocab=ELEMENT_VOCAB,
            lattice_stats=LATTICE_STATS,
            num_quant_bins=1024,
            sorting_order=SortingOrder.SPECIES,
            sequence_order=SequenceOrder.ATOMS_FIRST,
        )
    else:
        print("Loading tokenizer from model")
        tokenizer = ckpt.get("tokenizer")
        tokenizer = CrystalTokenizer.from_dict(tokenizer)

        so = getattr(tokenizer, "sequence_order", None)
        if so is None:
            # Kind of a fix for older versions of the tokenizer
            tokenizer = CrystalTokenizer(
                element_vocab=ELEMENT_VOCAB,
                lattice_stats=LATTICE_STATS,
                num_quant_bins=1024,
                sorting_order=tokenizer.sorting_order,
                sequence_order=getattr(
                    tokenizer, "sequence_order", SequenceOrder.ATOMS_FIRST
                ),
            )

    model = model.to(DEVICE)
    print("ROPE MODE", model.params.rope_mode)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters {num_parameters:.3f}")
    return model, tokenizer, ckpt_path, DEVICE


if __name__ == "__main__":
    
    import argparse
    from itertools import product
    import wandb
    import torch.multiprocessing as mp
    
    torch.set_num_threads(8)
    # Force 'spawn' method before any other logic
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass

    # ... now your argparse and dataset logic ...
    

    parser = argparse.ArgumentParser(description="Run LLaMa Transformer Train")
    parser.add_argument(
        "--oxidation_mode",
        type=str,
        default="bertos",
        help="select oxidation mode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="select oxidation mode",
    )
    parser.add_argument(
        "--check_dir",
        type=str,
        default="llm_bertos",
        help="select check point directory",
    )
    
    parser.add_argument(
        "--check_file",
        type=str,
        default=None,
        help="select check point",
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="select check point",
    )
    
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="alex_mp_20",
        help="select dataset",
    )
    args = parser.parse_args()
    
    

    
    ox_mode = args.oxidation_mode
    epoch = args.epochs
    checkpoint_dir = args.check_dir
    dataset_name = args.dataset
    torch.set_num_threads(8)

    ELEMENT_VOCAB = [Element.from_Z(i).symbol for i in range(1, 100)]
    LATTICE_STATS = {
        "a": (2.0, 10.0),
        "b": (2.0, 12.5),
        "c": (2.0, 20.0),
        "alpha": (60.0, 120.0),
        "beta": (60.0, 120.0),
        "gamma": (60.0, 120.0),
    }
    BATCH_SIZE = args.batch
    
    print('---------------Batch---------------------')
    print(BATCH_SIZE)
    cwd = os.path.join(os.path.dirname(__file__), "data")

    train_path = os.path.join(cwd, dataset_name, "train")
    train_dataset = MatterGenDataset(
        train_path, recalculate_cache=False, lattice_scaler=None
    )
    test_path = os.path.join(cwd, dataset_name, "val")
    test_dataset = MatterGenDataset(
        test_path, recalculate_cache=False, lattice_scaler=None
    )

    # ELEMENT_VOCAB, LATTICE_STATS = calculate_stats_from_dataset(train_dataset)

    tokenizer = CrystalTokenizer(
        element_vocab=ELEMENT_VOCAB,
        lattice_stats=LATTICE_STATS,
        num_quant_bins=1024,
        sorting_order=SortingOrder.REVERSE_SPECIES,  # SortingOrder.SPECIES,
        sequence_order=SequenceOrder.ATOMS_FIRST,
        oxidation_mode = ox_mode,
    )
    print("TOKENIZER SORTING ORDER", tokenizer.sorting_order, tokenizer.sequence_order)
    train_crystals = CrystalDataset(
        train_dataset,
        tokenizer,
        cache_path=f"v3_oxy_crystal_token_cache_latticelast_{dataset_name}_{ox_mode}_train_so{tokenizer.sorting_order.value}_seq{tokenizer.sequence_order}.joblib",
        recalculate_cache=False,
    )
    test_crystals = CrystalDataset(
        test_dataset,
        tokenizer,
        cache_path=f"v3_oxy_crystal_token_cache_latticelast_{dataset_name}_{ox_mode}_test_so{tokenizer.sorting_order.value}_seq{tokenizer.sequence_order}.joblib",
        recalculate_cache=False,
    )

    train_loader = DataLoader(
        train_crystals,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: pad_collate_fn(
            b, pad_token_id=tokenizer._special_to_id["[PAD]"]
        ),
        num_workers=2,
    )
    test_loader = DataLoader(
        test_crystals,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: pad_collate_fn(
            b, pad_token_id=tokenizer._special_to_id["[PAD]"]
        ),
        num_workers=2,
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim = 512
    condition_cfg = {
        "space_group": ConditionConfig(
            proj_layer_type="embedding", input_dim=230, out_dim=hidden_dim
        ),
        "reduced_formula": ConditionConfig(
            proj_layer_type="formula", input_dim=20, out_dim=hidden_dim
        ),
        "density": ConditionConfig(
            proj_layer_type="linear", input_dim=1, out_dim=hidden_dim
        ),
        "band_gap": ConditionConfig(
            proj_layer_type="linear", input_dim=1, out_dim=hidden_dim
        ),
        "bulk_modulus": ConditionConfig(
            proj_layer_type="linear", input_dim=1, out_dim=hidden_dim
        ),
        "hhi": ConditionConfig(
            proj_layer_type="linear", input_dim=1, out_dim=hidden_dim
        ),
        "mag_density": ConditionConfig(
            proj_layer_type="linear", input_dim=1, out_dim=hidden_dim
        ),
    }
    
    # 1. Use your load_model function to initialize
    # This handles model weights, tokenizer reconstruction, and device placement
    

    
    if args.check_file is not None:
        model, tokenizer, full_ckpt_path, DEVICE = load_model(args)
        
        # 2. Extract training state from the loaded checkpoint
        # We look for the raw dictionary returned by LLamaTransformer.load
        # (Note: In your load_model, LLamaTransformer.load returns (model, ckpt))
        checkpoint_data = torch.load(os.path.join(full_ckpt_path, args.check_file), map_location=DEVICE)
        start_epoch = checkpoint_data.get("epoch", -1) + 1
        best_val_loss = checkpoint_data.get("loss", 1e10)
        
        print('-------------------Start Epoch & Best Val Loss-------------------')
        print(start_epoch)
        print(best_val_loss)
        print('-----------------------------------------------------------------')
        # 4. Configure Optimizer and Restore State
        optimizer = configure_optimizer(model, type="adamw")
        if "optimizer" in checkpoint_data:
            print("Restoring optimizer state...")
            optimizer.load_state_dict(checkpoint_data["optimizer"])
    else:
        
        print('----------------------No checkpoint provided----------------------------')

        model_params = ModelArgs(
            dim=hidden_dim,
            n_layers=12,
            n_heads=16,
            vocab_size=len(tokenizer),
            multiple_of=256,
            max_seq_len=192,
            pad_id=tokenizer._special_to_id["[PAD]"],
            condition_config=condition_cfg,
            # rope_mode = RopeMode.BLOCKWISE,
            atoms_token_id=tokenizer._special_to_id["[ATOMS]"],
            lattice_token_id=tokenizer._special_to_id["[LATTICE]"],
            atom_block_size=4,
        )
        
            
        # Set the wandb project where this run will be logged.
    
    
        start_epoch = 0
        model = LLamaTransformer(model_params, loss_weights=None).to(DEVICE)
        ckpt_path = os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
            # f"llm_v3_LARGERopeStand_labelsmoothing=0.0_so{tokenizer.sorting_order.value}_seqo{tokenizer.sequence_order.value}_material_llama_adamw_condition_latticelast_dim{hidden_dim}",
            f"{checkpoint_dir}_{tokenizer.sorting_order.value}_{tokenizer.sequence_order.value}_material_llama_adamw_condition_latticelast_dim{hidden_dim}",
            
        )
    
        # ckpt = torch.load(os.path.join(ckpt_path, "model_18_0.907.pt"), map_location=DEVICE)
        # model.load_state_dict(ckpt["state_dict"], strict=False)
        # start_epoch = ckpt["epoch"] + 1
    
        num_parameters = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters {num_parameters:.3f}")
    
        optimizer = configure_optimizer(model, type="adamw")
        # optimizer.load_state_dict(ckpt["optimizer"])
        # model = torch.compile(model) # This actually is slower?
        
    wandb.login(key="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", relogin=True)
    wandb.init(entity="dipannoydip",project="materium_bertos", config=model_params)
    wandb.define_metric("epoch") # Define the epoch metric
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

    train_model(
        model,
        tokenizer,
        train_loader,
        test_loader,
        optimizer,
        epochs=epoch,
        device=DEVICE,
        ckpt_path=ckpt_path,
        start_epoch=start_epoch,
    )

    RELAX_GENERATED_MATERIALS = False
    num_total_samples = 32
    all_generated = []
    invalid_structures = 0
    for gen_idx in range(num_total_samples):
        try:
            generated_material = generate_structure(
                model, tokenizer, device=DEVICE, temperature=1.0
            )
            generated_material: Structure = tokenizer.detokenize(generated_material)
            save_struct_path = os.path.join(ckpt_path, "generated")
            os.makedirs(save_struct_path, exist_ok=True)
            cif_path = os.path.join(save_struct_path, f"material_sample_{gen_idx}.cif")
            generated_material.to(cif_path, "cif")
            print("Saved structure to", cif_path)
            all_generated.append(generated_material)
        except Exception as e:
            print("Invalid structure")
            invalid_structures += 1

    print(f"Invalid structures {invalid_structures}/{num_total_samples}")

    if RELAX_GENERATED_MATERIALS:

        rmsds = []
        relaxed_structures = []
        relaxed_energies = []
        for structure in all_generated:
            try:
                final_structure, final_energy, final_energy_per_atom, final_forces = (
                    relax_structure_mattersim(
                        structure, fmax=0.1, device="cpu", constrain_symmetry=True
                    )
                )
                relaxed_structures.append(final_structure)
                relaxed_energies.append(final_energy)
                print(f"The final energy is {final_energy_per_atom:.3f} eV per atom.")
                difference = abs(
                    structure.lattice.volume / final_structure.lattice.volume
                )
                print(f"Difference Volume: {difference:.3f} Å")
                print(
                    f"Structure {structure.get_space_group_info()} Final structure Space group: {final_structure.get_space_group_info()}"
                )
                print(f"Reduced formula {final_structure.composition.reduced_formula}")

                def print_lattice(s):
                    print(
                        f"Lattice: a={s.lattice.a:.3f}, b={s.lattice.b:.3f}, c={s.lattice.c:.3f}, "
                        f"alpha={s.lattice.alpha:.2f}, beta={s.lattice.beta:.2f}, gamma={s.lattice.gamma:.2f}"
                    )
                    print(f"Volume: {s.volume:.3f}")

                print_lattice(structure)
                print_lattice(final_structure)
                rmsd, found_match = compute_rmsd_angstrom(structure, final_structure)
                print("RMSD Angström", rmsd, "Found match", found_match)
                rmsds.append(rmsd)
                # structure_symm_rmsd(structure, final_structure)

                print(
                    "MSE CART:",
                    np.sqrt(
                        (structure.cart_coords - final_structure.cart_coords) ** 2
                    ).mean(),
                )

            except Exception as e:
                print("Weird error", e)

        print("Mean rmsd", np.mean(rmsds))
        eval_output = evaluate(
            relaxed_structures,
            energies=relaxed_energies,
            device=DEVICE,
            save_as=f"RESULT_MATTERGENEVALUATE_llm_{num_total_samples}",
            relax=False,
        )
        print("Mattergen eval:")
        print(eval_output)

