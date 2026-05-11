import math
import struct
import inspect
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict
from enum import Enum
from torch_scatter import scatter_add


@dataclass
class ConditionConfig:
    """Configuration for a single conditional input."""

    proj_layer_type: str
    input_dim: int
    out_dim: int

    def __post_init__(self):
        if self.proj_layer_type not in ["linear", "embedding", "formula", "sg_fingerprint"]:
            raise ValueError(f"Unknown proj_layer_type: {self.proj_layer_type}")


class RopeMode(Enum):
    STANDARD = "standard"
    BLOCKWISE = "blockwise"


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.1
    pad_id: int = -1
    condition_config: Optional[Dict[str, ConditionConfig]] = None
    rope_mode: RopeMode = RopeMode.STANDARD
    atoms_token_id: Optional[int] = None
    lattice_token_id: Optional[int] = None
    atom_block_size: int = 4

    def to_dict(self) -> Dict:
        """
        Serializes the ModelArgs instance to a dictionary.

        This method handles nested dataclasses and Enums correctly,
        making the output dictionary suitable for JSON serialization.
        """
        data = asdict(self)

        if "rope_mode" in data and isinstance(data["rope_mode"], RopeMode):
            data["rope_mode"] = data["rope_mode"].name

        return data
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelArgs":
        """
        Deserializes a dictionary into a ModelArgs instance.

        This method correctly reconstructs nested dataclasses and Enums
        from the dictionary's contents.
        """
        # If config_dict is already a ModelArgs object, return it. 
        # Otherwise, ensure we work with a dictionary.
        if isinstance(config_dict, cls):
            return config_dict
        
        config = config_dict.__dict__.copy() if hasattr(config_dict, "__dict__") else config_dict.copy()

        if "condition_config" in config and config["condition_config"] is not None:
            hydrated_conditions = {
                key: value if isinstance(value, ConditionConfig) else ConditionConfig(**value)
                for key, value in config["condition_config"].items()
            }
            config["condition_config"] = hydrated_conditions

        if "rope_mode" in config and isinstance(config["rope_mode"], str):
            try:
                config["rope_mode"] = RopeMode[config["rope_mode"]]
            except KeyError:
                raise ValueError(f"'{config['rope_mode']}' is not a valid RopeMode.")

        return cls(**config)

    # @classmethod
    # def from_dict(cls, config_dict: Dict) -> "ModelArgs":
    #     """
    #     Deserializes a dictionary into a ModelArgs instance.

    #     This method correctly reconstructs nested dataclasses and Enums
    #     from the dictionary's contents.
    #     """
    #     # if isinstance(config_dict, cls):
    #     #     return config_dict
        
    #     # # If it's not a dictionary but has a __dict__ attribute, use that
    #     # if not isinstance(config_dict, dict) and hasattr(config_dict, "__dict__"):
    #     #     config = config_dict.__dict__.copy()
    #     # else:
    #     #     # Original logic for plain dictionaries
    #     #     config = config_dict.copy()
    #     config = config_dict.copy()

    #     if "condition_config" in config and config["condition_config"] is not None:
    #         hydrated_conditions = {
    #             key: ConditionConfig(**value)
    #             for key, value in config["condition_config"].items()
    #         }
    #         config["condition_config"] = hydrated_conditions

    #     if "rope_mode" in config and isinstance(config["rope_mode"], str):
    #         try:
    #             config["rope_mode"] = RopeMode[config["rope_mode"]]
    #         except KeyError:
    #             raise ValueError(f"'{config['rope_mode']}' is not a valid RopeMode.")

    #     return cls(**config)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    if freqs_cis.dim() == 2:
        # (seqlen, dim) -> (1, seqlen, 1, dim) to broadcast over batch and heads
        seqlen, dim = freqs_cis.shape
        return freqs_cis.view(1, seqlen, 1, dim)
    elif freqs_cis.dim() == 3:
        # (bsz, seqlen, dim) -> (bsz, seqlen, 1, dim) to broadcast over heads
        bsz, seqlen, dim = freqs_cis.shape
        assert bsz == x.shape[0] and seqlen == x.shape[1] and dim == x.shape[-1]
        return freqs_cis.view(bsz, seqlen, 1, dim)
    else:
        raise ValueError(f"Unexpected freqs_cis dim: {freqs_cis.dim()}")


import torch


def build_blockwise_position_ids(
    tokens: torch.Tensor,
    cond_len: int,
    atoms_token_id: int,
    lattice_token_id: int,
    block_size: int = 4,
) -> torch.Tensor:
    """
    tokens: (B, S) original token ids (without prepended condition embeddings)
    cond_len: number of condition tokens prepended to the sequence (0 if none)

    Returns: (B, cond_len + S) position ids where positions strictly between
             [ATOMS] and [LATTICE] repeat 0..block_size-1; outside remain
             standard monotonic positions.

    Generation-safe: if [LATTICE] is not present yet, applies block-wise up to
    the current sequence end; once [LATTICE] appears, block-wise stops there.
    """
    device = tokens.device
    B, S = tokens.shape
    total_len = cond_len + S

    pos_ids = (
        torch.arange(total_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(B, total_len)
        .clone()
    )

    j = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, S)
    big = torch.full((B, S), S, device=device, dtype=torch.long)

    atoms_mask = tokens == atoms_token_id
    lattice_mask = tokens == lattice_token_id
    atoms_first = torch.where(atoms_mask, j, big).min(dim=1).values
    lattice_first = torch.where(lattice_mask, j, big).min(dim=1).values

    atoms_present = atoms_first < S
    lattice_present = lattice_first < S

    wrong_order = lattice_present & (~atoms_present | (lattice_first <= atoms_first))
    if torch.any(wrong_order).item():
        bad_rows = torch.nonzero(wrong_order, as_tuple=False).flatten().tolist()
        raise ValueError(f"[LATTICE] appears before [ATOMS] in rows: {bad_rows}")

    start = torch.where(atoms_present, atoms_first + 1, torch.full_like(atoms_first, S))
    end = torch.where(lattice_present, lattice_first, torch.full_like(lattice_first, S))

    in_span = (
        (j >= start.unsqueeze(1)) & (j < end.unsqueeze(1)) & atoms_present.unsqueeze(1)
    )

    local = (j - start.unsqueeze(1)) % block_size + cond_len + start.unsqueeze(1)

    repl = cond_len + j

    repl = torch.where(in_span, local, repl)

    pos_ids[:, cond_len : cond_len + S] = repl

    return pos_ids


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = (
                scores + self.mask[:, :, :seqlen, :seqlen]
            )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


from torch.nn.utils.rnn import pad_sequence


class FormulaEmbedder(nn.Module):

    def __init__(self, hidden_size, atom_embedder: nn.Module):
        super().__init__()
        # We have a max of 20 atom materials in a dataset
        self.composition_num_atoms = nn.Embedding(21, hidden_size)
        self.atom_embedder = atom_embedder

        self.combined_emb = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, reduced_formula_dict: Dict[str, torch.Tensor]):

        num_atoms = reduced_formula_dict["composition_num_atoms"]
        num_atoms_emb = self.composition_num_atoms(num_atoms)
        tokens = reduced_formula_dict["composition_symbol_tokens"]
        atom_elements_emb = self.atom_embedder(tokens)
        out_emb = self.combined_emb(atom_elements_emb + num_atoms_emb)

        num_atoms_per_sample = reduced_formula_dict["num_atoms_per_sample"]
        atom_sequences_list = torch.split(out_emb, num_atoms_per_sample.tolist())

        # `pad_sequence` takes this list of variable-length sequences and pads them
        # to the length of the longest sequence in the batch.
        # Shape: (batch_size, max_len_in_this_batch, hidden_size)
        padded_atom_sequences = pad_sequence(
            atom_sequences_list,
            batch_first=True,
            padding_value=0.0,
        )

        return padded_atom_sequences


def _compute_sg_fingerprints() -> np.ndarray:
    """
    Precompute 16-dim binary fingerprints for all 230 space groups.
    Called once during SpaceGroupEmbedding.__init__ to build the buffer.

    Feature layout:
      0-6  : crystal system one-hot (triclinic … cubic)
      7    : centrosymmetric (has inversion centre)
      8-11 : lattice centring one-hot (P / I / F / R; C/A/B → all zero)
      12   : has 2-fold rotation axis  (SG ≥ 3)
      13   : has 3-fold rotation axis  (SG ≥ 143)
      14   : has 4-fold rotation axis  (tetragonal + cubic)
      15   : has 6-fold rotation axis  (hexagonal)
    """
    from pymatgen.symmetry.groups import SpaceGroup as PmgSG

    # Laue-class centrosymmetric SG numbers (from crystallographic tables)
    _centrosymmetric = {
        2, 10, 11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
        74, 83, 84, 85, 86, 87, 88, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 147, 148,
        162, 163, 164, 165, 166, 167, 175, 176, 191, 192, 193, 194, 200, 201,
        202, 203, 204, 205, 206, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
    }
    _systems = ["triclinic", "monoclinic", "orthorhombic",
                "tetragonal", "trigonal", "hexagonal", "cubic"]

    fingerprints = np.zeros((230, 16), dtype=np.float32)
    for sg_num in range(1, 231):
        sg = PmgSG.from_int_number(sg_num)
        feat = np.zeros(16, dtype=np.float32)

        # Bits 0-6: crystal system (one-hot)
        sys_name = sg.crystal_system.lower()
        if sys_name in _systems:
            feat[_systems.index(sys_name)] = 1.0

        # Bit 7: centrosymmetric
        feat[7] = float(sg_num in _centrosymmetric)

        # Bits 8-11: lattice centring derived from HM symbol first letter
        centring = sg.symbol.strip()[0]
        if centring == "P":
            feat[8] = 1.0
        elif centring == "I":
            feat[9] = 1.0
        elif centring == "F":
            feat[10] = 1.0
        elif centring == "R":
            feat[11] = 1.0
        # C/A/B base-centred: all four bits stay 0

        # Bits 12-15: highest rotation axis present
        feat[12] = float(sg_num >= 3)                                       # 2-fold
        feat[13] = float(sg_num >= 143)                                     # 3-fold
        feat[14] = float((75 <= sg_num <= 142) or (sg_num >= 195))         # 4-fold
        feat[15] = float(168 <= sg_num <= 194)                             # 6-fold

        fingerprints[sg_num - 1] = feat
    return fingerprints


class SpaceGroupEmbedding(nn.Module):
    """
    Symmetry-aware space group embedding that replaces plain nn.Embedding(230, dim).

    A 16-dim crystallographic fingerprint (crystal system, centring, centrosymmetry,
    rotation axes) is projected into the model's hidden dimension via a small MLP,
    giving the model a meaningful geometric prior rather than an arbitrary random init.

    Input / output shapes are identical to nn.Embedding, making this a drop-in
    replacement inside append_condition:
      Input:  (B, SeqLen) integer tensor, values in 1–230
      Output: (B, SeqLen, hidden_dim)

    Migration from a checkpoint trained with nn.Embedding:
        LLamaTransformer.migrate_to_sg_fingerprint(src_ckpt, dst_ckpt)
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        fps = _compute_sg_fingerprints()          # (230, 16), computed once
        self.register_buffer("fingerprints", torch.tensor(fps, dtype=torch.float32))
        self.projector = nn.Sequential(
            nn.Linear(16, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
        )

    def forward(self, sg_numbers: torch.Tensor) -> torch.Tensor:
        orig_shape = sg_numbers.shape          # (B,) or (B, SeqLen)
        flat = sg_numbers.reshape(-1)          # (N,)
        fps = self.fingerprints[flat - 1]      # (N, 16)  — 1-indexed→0-indexed
        out = self.projector(fps)              # (N, hidden_dim)
        return out.reshape(*orig_shape, self.hidden_dim)  # (B [, SeqLen], hidden_dim)


class LLamaTransformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs, loss_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.pad_id = params.pad_id
        self.cond_cfg = self.params.condition_config

        self.register_buffer("loss_weights", loss_weights)

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = (
            self.output.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.cond_proj_layers = nn.ModuleDict()
        self.cond_to_type_id = {}
        # Type ID 0 is reserved for the main crystal tokens
        self.token_type_id = 0
        self.condition_mask_embeddings = nn.ParameterDict()

        if params.condition_config:
            # Assign a unique type ID to each condition, starting from 1
            for i, (name, cfg) in enumerate(params.condition_config.items()):
                type_id = i + 1  # 0 is for tokens
                self.cond_to_type_id[name] = type_id

                if cfg.proj_layer_type.lower() == "linear":
                    self.cond_proj_layers[name] = nn.Linear(
                        cfg.input_dim, params.dim, bias=False
                    )
                elif cfg.proj_layer_type.lower() == "embedding":
                    self.cond_proj_layers[name] = nn.Embedding(
                        cfg.input_dim, params.dim
                    )
                elif cfg.proj_layer_type.lower() == "formula":
                    self.cond_proj_layers[name] = FormulaEmbedder(
                        params.dim, self.tok_embeddings
                    )
                elif cfg.proj_layer_type.lower() == "sg_fingerprint":
                    self.cond_proj_layers[name] = SpaceGroupEmbedding(hidden_dim=params.dim)
                else:
                    raise ValueError("No layer of type", cfg.proj_layer_type, "found")

                self.condition_mask_embeddings[name] = nn.Parameter(
                    torch.randn(1, 1, params.dim)
                )
            n_cond_types = len(params.condition_config) + 1
            self.type_embeddings = nn.Embedding(n_cond_types, params.dim)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers)
                )

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_positional_encodings(
        self, tokens: torch.Tensor, cond_len: int, final_seq_len: int
    ):
        if self.params.rope_mode == RopeMode.BLOCKWISE:
            assert (
                self.params.atoms_token_id is not None
                and self.params.lattice_token_id is not None
            ), "atoms_token_id and lattice_token_id must be set for blockwise RoPE."
            pos_ids = build_blockwise_position_ids(
                tokens=tokens,
                cond_len=cond_len,
                atoms_token_id=self.params.atoms_token_id,
                lattice_token_id=self.params.lattice_token_id,
                block_size=self.params.atom_block_size,
            )  # shape (B, final_seq_len)
            freqs_cos = self.freqs_cos[pos_ids]  # (B, final_seq_len, head_dim/2)
            freqs_sin = self.freqs_sin[pos_ids]  # (B, final_seq_len, head_dim/2)
        else:
            freqs_cos = self.freqs_cos[:final_seq_len]  # (final_seq_len, head_dim/2)
            freqs_sin = self.freqs_sin[:final_seq_len]  # (final_seq_len, head_dim/2

        return freqs_cos, freqs_sin

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = {},
        return_hidden_states: bool = False,
        position_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if conditions is not None:
            h, cond_len = self.append_condition(tokens, conditions, _bsz, h)

        h = self.dropout(h)
        _, final_seq_len, _ = h.shape

        freqs_cos, freqs_sin = self.get_positional_encodings(
            tokens, cond_len, final_seq_len
        )

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if return_hidden_states:
            logits = self.output(h[:, -seqlen:, :])
            return logits, h

        if targets is not None:
            logits = self.output(
                h[:, -seqlen:, :]
            )  # Only calculate for the actual sequence not the conditions

            if position_weights is not None:
                # Per-position weighted loss: upweights lattice tokens, element tokens, etc.
                per_tok = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.pad_id,
                    weight=self.loss_weights,
                    label_smoothing=0.0,
                    reduction="none",
                )
                pad_mask = (targets.reshape(-1) != self.pad_id).float()
                pw = position_weights.reshape(-1) * pad_mask
                self.last_loss = (per_tok * pw).sum() / pw.sum().clamp(min=1)
            else:
                self.last_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.pad_id,
                    weight=self.loss_weights,
                    label_smoothing=0.0,
                )
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(
                h[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def append_condition(self, tokens, conditions, _bsz, h):
        token_type_ids = torch.full_like(tokens, self.token_type_id)

        prepended_embs = []
        prepended_type_ids = []

        for name, tensor in conditions.items():
            if name.endswith("_mask"):
                continue

            if name in self.cond_proj_layers:
                # Project condition to model dimension
                cond_emb = self.cond_proj_layers[name](tensor)
                mask_key = f"{name}_mask"
                if mask_key in conditions:
                    mask = conditions[mask_key]  # Shape: [B]

                    mask_emb = self.condition_mask_embeddings[name]

                    broadcast_mask = mask.view(_bsz, 1, 1)

                    cond_emb = torch.where(broadcast_mask, mask_emb, cond_emb)

                prepended_embs.append(cond_emb)

                type_id = self.cond_to_type_id[name]
                cond_type_tensor = torch.full(
                    cond_emb.shape[:2], type_id, device=tokens.device, dtype=torch.long
                )
                prepended_type_ids.append(cond_type_tensor)

            # Sequence is [cond1, cond2, ..., tokens]
        if len(prepended_embs) != 0:
            h = torch.cat(prepended_embs + [h], dim=1)
            cond_len = sum(e.shape[1] for e in prepended_embs)

            all_type_ids = torch.cat(prepended_type_ids + [token_type_ids], dim=1)
            h = h + self.type_embeddings(all_type_ids)
        else:
            cond_len = 0
            h = h + self.type_embeddings(token_type_ids)
        return h, cond_len

    def save(self, path: str, **kwargs):
        torch.save(
            {"state_dict": self.state_dict(), "model_config": self.params, **kwargs},
            path,
        )

    @staticmethod
    def load(path: str, strict=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        conf_data = checkpoint["model_config"]
        # if not isinstance(conf_data, dict) and hasattr(conf_data, "__dict__"):
        #     conf_data = conf_data.__dict__
        model_config = ModelArgs.from_dict(conf_data)
        # model_config = ModelArgs.from_dict(checkpoint["model_config"])
        model = LLamaTransformer(params=model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        del checkpoint["state_dict"]
        return model, checkpoint

    @staticmethod
    def migrate_to_sg_fingerprint(
        src_ckpt_path: str,
        dst_ckpt_path: str,
        condition_name: str = "space_group",
        n_fit_steps: int = 500,
        fit_lr: float = 1e-3,
    ) -> None:
        """
        One-time migration: loads a checkpoint whose 'space_group' condition was trained
        with nn.Embedding, replaces it with SpaceGroupEmbedding (warm-started to
        approximate the old embedding weights), and saves a new checkpoint.

        After running this once you can fine-tune the model normally — no retraining
        from scratch is required.  Only the ~34 K projector parameters are uninitialised
        (they are warm-started here); all other weights transfer exactly.

        Usage:
            LLamaTransformer.migrate_to_sg_fingerprint(
                "checkpoints/old_model.pt",
                "checkpoints/sg_fingerprint_model.pt",
            )
        """
        print(f"Loading source checkpoint: {src_ckpt_path}")
        ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)

        # --- Build old model and extract its SG embedding weights ---
        old_config = ModelArgs.from_dict(ckpt["model_config"])
        old_model = LLamaTransformer(params=old_config)
        old_model.load_state_dict(ckpt["state_dict"], strict=True)

        if condition_name not in old_model.cond_proj_layers:
            raise ValueError(
                f"Condition '{condition_name}' not found in checkpoint. "
                f"Available: {list(old_model.cond_proj_layers.keys())}"
            )
        old_layer = old_model.cond_proj_layers[condition_name]
        if not isinstance(old_layer, nn.Embedding):
            raise ValueError(
                f"Expected nn.Embedding for '{condition_name}', got {type(old_layer)}. "
                "Model may already be migrated."
            )
        # (230, dim) — the trained SG embedding matrix we want to approximate
        old_sg_weight = old_layer.weight.data.clone()  # detach from old model
        del old_model

        # --- Build new config with sg_fingerprint ---
        new_config_dict = old_config.to_dict()
        new_config_dict["condition_config"][condition_name]["proj_layer_type"] = "sg_fingerprint"
        new_config = ModelArgs.from_dict(new_config_dict)

        # --- Build new model; load all weights except the replaced SG layer ---
        new_model = LLamaTransformer(params=new_config)
        missing, unexpected = new_model.load_state_dict(ckpt["state_dict"], strict=False)
        sg_missing = [k for k in missing if condition_name in k]
        print(f"SG embedding keys not loaded (expected): {sg_missing}")
        if unexpected:
            print(f"Unexpected keys (investigate): {unexpected}")

        # --- Warm-start: fit projector so MLP(fingerprint[i]) ≈ old_sg_weight[i] ---
        sg_emb: SpaceGroupEmbedding = new_model.cond_proj_layers[condition_name]
        optimizer = torch.optim.Adam(sg_emb.projector.parameters(), lr=fit_lr)
        sg_idx = torch.arange(1, 231, dtype=torch.long)  # (230,)
        sg_emb.train()
        print(f"Warm-starting SpaceGroupEmbedding projector for {n_fit_steps} steps...")
        for step in range(n_fit_steps):
            optimizer.zero_grad()
            fps = sg_emb.fingerprints[sg_idx - 1]   # (230, 16)
            pred = sg_emb.projector(fps)             # (230, hidden_dim)
            loss = F.mse_loss(pred, old_sg_weight)
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{n_fit_steps}: MSE = {loss.item():.6f}")
        sg_emb.eval()

        # --- Save migrated checkpoint ---
        new_ckpt = {k: v for k, v in ckpt.items() if k != "state_dict"}
        new_ckpt["state_dict"] = new_model.state_dict()
        new_ckpt["model_config"] = new_config
        torch.save(new_ckpt, dst_ckpt_path)
        print(f"Migrated checkpoint saved to: {dst_ckpt_path}")
