"""One description per crystal for the Wyckoff representation.

encode() writes whatever description spglib and pyxtal hand back, and a crystal
has several equally valid ones:

  setting   An origin shift or axis change in the group's normaliser maps the
            group onto itself but relabels positions. Rock-salt NaCl is both
            "Na 4a, Cl 4b" and "Na 4b, Cl 4a".

  point     Any orbit member of the position's generator form can carry the free
            parameters. The 24g orbit of Im-3 is (0, y, z) and also (0, -y, z).

  origin    A polar group (P6_3mc, Pmn2_1, P1, ...) can be shifted continuously
            along its polar axes, so the free coordinates along them are only
            defined up to a common offset.

Training on that mixture teaches the model several sequences for one crystal.
This module picks one, by a rule that depends only on the crystal:

  1. Orbits are ordered by Pauling electronegativity, ascending; ties by Wyckoff
     letter, then atomic number, species token and free-parameter bins.
  2. Each orbit is written at the member of its generator form with the smallest
     free-parameter bins.
  3. Every normaliser coset representative is tried (pyxtal's Wyckoff-set table),
     and for a polar group every origin that puts the first orbit at zero along
     the polar axes. The description whose sequence is lexicographically smallest
     is the one written.

A candidate is only accepted if its orbits expand to exactly the atoms of the
crystal and its lattice obeys the crystal system's constraints, so whatever is
chosen decodes to the same structure as the description encode() produced.
"""
import math
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Element, Lattice
from pymatgen.core.operations import SymmOp

from materium.wyckoff import _free_indices, _group, _wrap_close, free_lattice_indices, full_lattice

SET_TOL = 1e-4


@lru_cache(maxsize=None)
def electronegativity(symbol: str) -> float:
    """Pauling electronegativity; elements without one (He, Ne, Ar) sort last."""
    x = Element(symbol).X
    return 10.0 if x is None or math.isnan(x) else float(x)


@lru_cache(maxsize=None)
def _wyckoff_sets():
    import os

    import pyxtal
    from monty.serialization import loadfn
    return loadfn(os.path.join(os.path.dirname(pyxtal.__file__), "database", "wyckoff_sets.json"))


@lru_cache(maxsize=None)
def normaliser_ops(number: int) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """(R, t) of every coset representative; identity first. Volume-changing ones
    are dropped, since they would not map the cell's atoms onto the same count."""
    ops = []
    for s in _wyckoff_sets()[str(number)]["Coset Representative"]:
        op = SymmOp.from_xyz_str(s)
        R = np.array(op.rotation_matrix, dtype=float)
        if abs(abs(np.linalg.det(R)) - 1.0) < 1e-6:
            ops.append((R, np.array(op.translation_vector, dtype=float)))
    return tuple(ops)


@lru_cache(maxsize=None)
def polar_axes(number: int) -> Tuple[int, ...]:
    """Coordinate axes every rotation of the group leaves fixed: the directions the
    origin can slide along. In standard settings these are coordinate axes."""
    rots = [np.array(op.rotation_matrix) for op in _group(number)[0].ops]
    out = []
    for d in range(3):
        e = np.zeros(3)
        e[d] = 1.0
        if all(np.allclose(R @ e, e) for R in rots):
            out.append(d)
    return tuple(out)


def _snap(p: np.ndarray) -> np.ndarray:
    p = np.mod(p, 1.0)
    p[p > 1.0 - 1e-8] = 0.0
    p[p < 1e-8] = 0.0
    return p


def _same_set(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) != len(b):
        return False
    d = np.abs(((a[:, None, :] - b[None, :, :]) + 0.5) % 1.0 - 0.5)
    close = np.all(d < SET_TOL, axis=2)
    return bool(close.any(axis=1).all() and close.any(axis=0).all())


def _expand(wp, p: np.ndarray) -> np.ndarray:
    pts = np.mod(np.array(wp.apply_ops(p), dtype=float), 1.0)
    kept = []
    for q in pts:
        if not any(_wrap_close(q, r, 1e-6) for r in kept):
            kept.append(q)
    return np.array(kept)


def _form_members(wp, pts: np.ndarray) -> np.ndarray:
    """Orbit members that lie on the position's generator form."""
    op0 = wp.ops[0]
    img = np.mod(pts @ np.array(op0.rotation_matrix).T + np.array(op0.translation_vector), 1.0)
    d = np.abs(((img - pts) + 0.5) % 1.0 - 0.5)
    return pts[np.all(d < SET_TOL, axis=1)]


def _identify(group, orbit: np.ndarray):
    """(wp index, form members) of the Wyckoff position this orbit occupies."""
    m = len(orbit)
    for k, wp in enumerate(group):
        if wp.multiplicity != m:
            continue
        members = _form_members(wp, orbit)
        if len(members) and _same_set(_expand(wp, members[0]), orbit):
            return k, members
    return None


def canonical_orbits(
    number: int,
    lattice: Lattice,
    frac: np.ndarray,
    orbit_of: Sequence[int],
    elements: Sequence[str],
    sids: Sequence[int],
    quantize: Callable[[float], int],
    lattice_bins: Callable[[Tuple[float, ...]], Optional[List[int]]],
):
    """The canonical description of a decoded standard cell.

    frac/orbit_of are the atoms of the conventional standard cell and the orbit
    each belongs to; elements/sids are per orbit. quantize maps a coordinate in
    [0, 1) to its bin, and lattice_bins maps (a, b, c, alpha, beta, gamma) to the
    free-parameter bins, or None if the cell cannot be tokenized.

    Returns (orbits, lattice) with orbits a list of (orbit index, wp index, free
    values) in canonical order, or None if no candidate passes the checks.
    """
    group = _group(number)
    n_wp = len(group)
    orbit_of = np.asarray(orbit_of)
    n_orbits = len(elements)
    orbits = [np.asarray(frac[orbit_of == j], dtype=float) for j in range(n_orbits)]
    head = [(electronegativity(elements[j]), Element(elements[j]).Z, sids[j]) for j in range(n_orbits)]
    polar = list(polar_axes(number))
    M = lattice.matrix

    best = None
    for R, t in normaliser_ops(number):
        # x' = R x + t, with Cartesian positions unchanged: M' = R^-T M.
        new_lat = Lattice(np.linalg.inv(R).T @ M)
        params = tuple(new_lat.parameters)
        full = full_lattice(number, [params[i] for i in free_lattice_indices(number)])
        if any(abs(full[i] - params[i]) > (1e-3 * params[i] if i < 3 else 0.05) for i in range(6)):
            continue
        lat_bins = lattice_bins(params)
        if lat_bins is None:
            continue

        placed = []
        for j in range(n_orbits):
            hit = _identify(group, np.mod(orbits[j] @ R.T + t, 1.0))
            if hit is None:
                break
            placed.append(hit)
        if len(placed) != n_orbits:
            continue

        def rank(j):
            e, z, s = head[j]
            return (e, n_wp - 1 - placed[j][0], z, s)

        # Origins: zero along the polar axes for every member of every orbit that
        # ranks first. The candidate set is defined by the crystal alone.
        shifts = [np.zeros(3)]
        if polar:
            first = min(rank(j) for j in range(n_orbits))
            shifts = []
            for j in range(n_orbits):
                if rank(j) != first:
                    continue
                for p in placed[j][1]:
                    s = np.zeros(3)
                    s[polar] = -p[polar]
                    shifts.append(s)

        for s in shifts:
            entries = []
            for j, (k, members) in enumerate(placed):
                free_idx = _free_indices(group[k])
                best_rep = None
                for p in _snap(members + s):
                    vals = tuple(float(p[i]) for i in free_idx)
                    bins = tuple(quantize(v) for v in vals)
                    if best_rep is None or bins < best_rep[0]:
                        best_rep = (bins, vals)
                entries.append((rank(j) + (best_rep[0],), j, k, list(best_rep[1])))
            entries.sort(key=lambda e: e[0])
            key = (tuple(e[0] for e in entries), tuple(lat_bins))
            if best is None or key < best[0]:
                best = (key, [(j, k, vals) for _, j, k, vals in entries], params)

    if best is None:
        return None
    return best[1], best[2]
