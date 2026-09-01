"""Geometry-aware loss for fractional-coordinate tokens.

Coordinate bins and lattice bins share the same token ids (both are
``quant_offset + bin``), so a coordinate-specific loss cannot identify its
targets from token ids alone.  It has to know each position's *role*, which is
purely structural: between ``[ATOMS]`` and ``[LATTICE]`` the sequence repeats a
fixed block of ``block_size`` tokens, and the coordinate slots sit at known
offsets inside that block.

What this module provides:

* :func:`build_token_roles` -- per-position role + coordinate axis (x/y/z).
* :func:`circular_soft_targets` -- a wrapped-Gaussian target kernel over bins,
  using circular distance so bin 0 and bin B-1 are neighbours.
* :func:`apply_translation_augmentation` -- a global fractional translation
  applied directly in token space (exact, no requantisation).
* :func:`structured_token_loss` -- the composed loss plus per-group metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# Position roles. PAD is excluded from every loss term.
ROLE_PAD = 0
ROLE_SPECIAL = 1
ROLE_SPECIES = 2
ROLE_COORD = 3
ROLE_LATTICE = 4


@dataclass
class TokenLayout:
    """Everything needed to recover token roles from a padded token tensor."""

    atoms_token_id: int
    lattice_token_id: int
    pad_id: int
    quant_offset: int
    num_quant_bins: int
    block_size: int = 4
    coords_first: bool = False


@dataclass
class CoordLossConfig:
    """Configuration for the geometry-aware coordinate loss.

    sigma_bins:   width of the wrapped-Gaussian target, in bins. 0 disables
                  smoothing and falls back to one-hot targets.
    coord_weight: weight of the coordinate term relative to the rest.
    hard_mix:     blend back toward plain one-hot CE on coordinates
                  (0.0 = fully soft, 1.0 = original behaviour).
    """

    sigma_bins: float = 2.0
    coord_weight: float = 1.0
    hard_mix: float = 0.0


def build_token_roles(
    tokens: torch.Tensor, layout: TokenLayout
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify every position of a padded token tensor.

    Args:
        tokens: (B, S) token ids, padded with ``layout.pad_id``.

    Returns:
        roles:      (B, S) one of the ROLE_* constants.
        coord_axis: (B, S) 0/1/2 for x/y/z at coordinate positions, -1 elsewhere.

    Mirrors the span logic of ``build_blockwise_position_ids`` and is likewise
    generation-safe: if ``[LATTICE]`` has not been emitted yet, the atom block
    is taken to run to the end of the sequence.
    """
    bsz, seqlen = tokens.shape
    device = tokens.device

    j = torch.arange(seqlen, device=device, dtype=torch.long)
    j = j.unsqueeze(0).expand(bsz, seqlen)
    big = torch.full((bsz, seqlen), seqlen, device=device, dtype=torch.long)

    atoms_first = torch.where(tokens == layout.atoms_token_id, j, big).min(dim=1).values
    lattice_first = (
        torch.where(tokens == layout.lattice_token_id, j, big).min(dim=1).values
    )

    # First atom-block token; seqlen (i.e. empty span) when [ATOMS] is absent.
    start = torch.where(
        atoms_first < seqlen, atoms_first + 1, torch.full_like(atoms_first, seqlen)
    )
    end = lattice_first  # already seqlen when [LATTICE] is absent

    in_atoms = (j >= start.unsqueeze(1)) & (j < end.unsqueeze(1))
    offset = (j - start.unsqueeze(1)) % layout.block_size

    is_quant = (tokens >= layout.quant_offset) & (
        tokens < layout.quant_offset + layout.num_quant_bins
    )

    if layout.coords_first:
        # block = [x, y, z, elem]
        coord_slot = offset < 3
        species_slot = offset == 3
        axis = offset
    else:
        # block = [elem, x, y, z]
        coord_slot = offset >= 1
        species_slot = offset == 0
        axis = offset - 1

    is_coord = in_atoms & coord_slot & is_quant
    is_species = in_atoms & species_slot
    is_lattice = (j > lattice_first.unsqueeze(1)) & is_quant

    roles = torch.full((bsz, seqlen), ROLE_SPECIAL, device=device, dtype=torch.long)
    roles.masked_fill_(is_lattice, ROLE_LATTICE)
    roles.masked_fill_(is_species, ROLE_SPECIES)
    roles.masked_fill_(is_coord, ROLE_COORD)
    roles.masked_fill_(tokens == layout.pad_id, ROLE_PAD)

    coord_axis = torch.full((bsz, seqlen), -1, device=device, dtype=torch.long)
    coord_axis = torch.where(roles == ROLE_COORD, axis, coord_axis)

    return roles, coord_axis


def circular_bin_distance(num_bins: int, device=None) -> torch.Tensor:
    """(B, B) matrix of circular distances between bin indices."""
    idx = torch.arange(num_bins, device=device, dtype=torch.long)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return torch.minimum(d, num_bins - d)


def circular_soft_targets(
    num_bins: int, sigma_bins: float, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Row t is the target distribution for true bin t.

    A wrapped Gaussian discretised onto bins: mass decays with *circular*
    distance, so bin 0 and bin B-1 are adjacent and a near-miss is cheap.
    ``sigma_bins <= 0`` returns one-hot rows.
    """
    if sigma_bins <= 0:
        return torch.eye(num_bins, device=device, dtype=dtype)
    d = circular_bin_distance(num_bins, device=device).to(dtype)
    return torch.softmax(-(d**2) / (2.0 * float(sigma_bins) ** 2), dim=-1)


def soft_target_entropy(kernel: torch.Tensor) -> float:
    """Irreducible loss floor of the soft targets, in nats.

    Every row has the same entropy by circular symmetry, so row 0 suffices.
    A perfect model reaches this value, not 0.
    """
    p = kernel[0]
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def apply_translation_augmentation(
    tokens: torch.Tensor,
    roles: torch.Tensor,
    coord_axis: torch.Tensor,
    layout: TokenLayout,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply a random global translation to each structure, in token space.

    A crystal is defined only up to a choice of origin, but the tokenizer emits
    whatever origin the source database happened to store, so each structure has
    infinitely many equally-correct token sequences and the model is forced to
    memorise one arbitrary representative.  Translating by an exact multiple of
    the bin width is a cyclic shift of the bin index, so this is exact -- no
    requantisation error -- and needs no re-tokenisation, which matters because
    the token cache is expensive to rebuild.

    Only coordinate tokens move.  Lattice parameters are not translationally
    related and are left untouched.
    """
    bsz = tokens.shape[0]
    shifts = torch.randint(
        0,
        layout.num_quant_bins,
        (bsz, 3),
        generator=generator,
        device=tokens.device,
        dtype=torch.long,
    )

    axis = coord_axis.clamp_min(0)  # -1 -> 0 at non-coord slots (masked out below)
    per_token_shift = shifts.gather(1, axis)

    bins = tokens - layout.quant_offset
    shifted = (bins + per_token_shift) % layout.num_quant_bins + layout.quant_offset

    return torch.where(roles == ROLE_COORD, shifted, tokens)


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean of ``values`` over ``mask``; 0 when the mask is empty."""
    m = mask.to(values.dtype)
    if weights is not None:
        m = m * weights
    denom = m.sum()
    return (values * m).sum() / denom.clamp_min(1e-8)


def structured_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    roles: torch.Tensor,
    kernel: torch.Tensor,
    layout: TokenLayout,
    config: CoordLossConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cross-entropy everywhere, geometry-aware soft targets on coordinates.

    Coordinate and non-coordinate terms are normalised by their own token counts
    rather than pooled into one mean, so their relative weight does not drift
    with the number of atoms in a batch.

    Returns the scalar loss and a dict of detached per-group metrics.
    """
    vocab = logits.size(-1)
    logits = logits.reshape(-1, vocab).float()
    targets = targets.reshape(-1)
    roles = roles.reshape(-1)

    logp = F.log_softmax(logits, dim=-1)

    # Full-vocabulary NLL at every position; the group masks select from it.
    safe_targets = targets.clamp(0, vocab - 1)
    nll = -logp.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

    is_coord = roles == ROLE_COORD
    is_species = roles == ROLE_SPECIES
    is_lattice = roles == ROLE_LATTICE
    is_special = roles == ROLE_SPECIAL
    is_other = is_species | is_lattice | is_special

    weights = None
    if class_weights is not None:
        weights = class_weights.to(logp.dtype)[safe_targets]

    loss_other = _masked_mean(nll, is_other, weights)

    metrics: Dict[str, float] = {}
    zero = logp.new_zeros(())
    loss_coord = zero
    coord_idx = is_coord.nonzero(as_tuple=True)[0]

    if coord_idx.numel() > 0:
        qo, nbins = layout.quant_offset, layout.num_quant_bins
        # Restrict to the bin columns. log_softmax is still over the full
        # vocabulary, so probability mass placed on non-bin tokens is penalised.
        logp_bins = logp.index_select(0, coord_idx)[:, qo : qo + nbins]
        bins = (targets.index_select(0, coord_idx) - qo).clamp_(0, nbins - 1)

        soft = kernel.index_select(0, bins).to(logp_bins.dtype)
        loss_soft = -(soft * logp_bins).sum(dim=-1).mean()

        hard_nll = nll.index_select(0, coord_idx)
        loss_hard = hard_nll.mean()

        mix = float(config.hard_mix)
        loss_coord = (1.0 - mix) * loss_soft + mix * loss_hard

        with torch.no_grad():
            pred_bin = logp_bins.argmax(dim=-1)
            d = (pred_bin - bins).abs()
            d = torch.minimum(d, nbins - d)
            metrics["coord_soft"] = float(loss_soft)
            metrics["coord_ce_hard"] = float(loss_hard)
            # Mean absolute error in fractional units, wrapped across the cell.
            metrics["coord_mae_frac"] = float(d.float().mean()) / nbins
            metrics["coord_acc"] = float((d == 0).float().mean())
            # Fraction of coordinate slots whose top token is not a bin token.
            top = logp.index_select(0, coord_idx).argmax(dim=-1)
            off_grid = (top < qo) | (top >= qo + nbins)
            metrics["coord_off_grid"] = float(off_grid.float().mean())

    loss = loss_other + float(config.coord_weight) * loss_coord

    with torch.no_grad():
        metrics["loss_other"] = float(loss_other)
        if is_species.any():
            metrics["species_ce"] = float(_masked_mean(nll, is_species))
            pred = logp.argmax(dim=-1)
            metrics["species_acc"] = float(
                _masked_mean((pred == targets).to(logp.dtype), is_species)
            )
        if is_lattice.any():
            metrics["lattice_ce"] = float(_masked_mean(nll, is_lattice))
        if is_special.any():
            metrics["special_ce"] = float(_masked_mean(nll, is_special))

    return loss, metrics
