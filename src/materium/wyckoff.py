"""Wyckoff representation of a crystal: space group, orbits, free parameters.

The current tokenizer writes every atom's three fractional coordinates as
independent bins. A space group is a set of EXACT relations between those
coordinates (an atom at 1/4, a pair related by a mirror), and independent
per-coordinate bins have no way to express a relation, only to approximate each
number separately. Every measurement so far says that is where the model fails:
across all runs the structure rate barely moves while the space-group rate swings
by 14 points.

This representation removes the problem instead of training against it. A
structure is written as

    space group number
    for each symmetry orbit:  element, Wyckoff position, its FREE parameters only
    lattice:                  the free lattice parameters of the crystal system

and the full structure is recovered by applying the space group's operations to
each orbit's representative site. Symmetry then holds by construction, and a
special position such as 4a in Fm-3m carries no coordinates at all -- nothing to
get wrong.

pyxtal supplies the Wyckoff tables and the orbit operations; spglib, through
pyxtal, does the symmetry analysis. Everything is done in the space group's
standard setting, which is what makes a Wyckoff letter meaningful.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure

# Same tolerance the match-rate script assigns space groups with, so the group a
# structure is encoded under is the group it will later be judged against.
SYMPREC = 0.1

# Atoms per lattice point in the conventional cell, keyed by the first letter of
# the Hermann-Mauguin symbol. R groups are taken in the hexagonal setting, which
# is what pyxtal standardises to, so they carry three lattice points.
_CENTERING = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}

# Primitive basis vectors, as rows in conventional fractional coordinates, for
# each centering. Each has determinant 1/centering. R is the obverse hexagonal
# setting, lattice points (0,0,0), (2/3,1/3,1/3), (1/3,2/3,2/3).
_PRIMITIVE_BASIS = {
    "A": [[1, 0, 0], [0, 1 / 2, 1 / 2], [0, -1 / 2, 1 / 2]],
    "B": [[1 / 2, 0, 1 / 2], [0, 1, 0], [-1 / 2, 0, 1 / 2]],
    "C": [[1 / 2, 1 / 2, 0], [-1 / 2, 1 / 2, 0], [0, 0, 1]],
    "I": [[-1 / 2, 1 / 2, 1 / 2], [1 / 2, -1 / 2, 1 / 2], [1 / 2, 1 / 2, -1 / 2]],
    "F": [[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [1 / 2, 1 / 2, 0]],
    "R": [[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]],
}


@dataclass
class WyckoffSite:
    element: str
    wp_index: int          # index into pyxtal's Wyckoff list for this group
    label: str             # e.g. "4a"; carried for readability only
    multiplicity: int
    free: List[float]      # the orbit's free parameters, in [0, 1)


@dataclass
class WyckoffCrystal:
    space_group: int
    lattice: Tuple[float, float, float, float, float, float]   # a b c alpha beta gamma
    sites: List[WyckoffSite] = field(default_factory=list)

    @property
    def num_atoms(self) -> int:
        return sum(s.multiplicity for s in self.sites)

    @property
    def num_free(self) -> int:
        return sum(len(s.free) for s in self.sites)


@lru_cache(maxsize=None)
def _group(number):
    # Constructing a pyxtal Group reads its tables; generation asks for the same
    # few groups thousands of times, so build each once.
    from pyxtal.symmetry import Group
    return Group(number)


def _free_indices(wp) -> List[int]:
    """Which of x, y, z this Wyckoff position leaves free.

    Read off the position's first operation: column j of its rotation part is
    non-zero exactly when coordinate j of the representative feeds into the
    result, i.e. when it is a free parameter. 4a in Fm-3m is (0,0,0) -- no free
    columns. 24e is (x,0,0) -- one. The general position is (x,y,z) -- three.
    """
    rot = np.array(wp.ops[0].rotation_matrix)
    return [j for j in range(3) if np.any(np.abs(rot[:, j]) > 1e-8)]


@lru_cache(maxsize=None)
def group_table(number) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """(multiplicity, free coordinate indices) for each Wyckoff index of a group."""
    return tuple((int(wp.multiplicity), tuple(_free_indices(wp))) for wp in _group(number))


@lru_cache(maxsize=None)
def centering(number) -> int:
    return _CENTERING[_group(number).symbol[0]]


def crystal_system(number) -> str:
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 194:
        return "hexagonal"      # trigonal included: hexagonal axes throughout
    return "cubic"


# Which of (a, b, c, alpha, beta, gamma) the crystal system leaves free. The rest
# are fixed by the system in the standard setting, so they carry no token.
_FREE_LATTICE = {
    "triclinic": (0, 1, 2, 3, 4, 5),
    "monoclinic": (0, 1, 2, 4),     # b-unique: alpha = gamma = 90
    "orthorhombic": (0, 1, 2),
    "tetragonal": (0, 2),
    "hexagonal": (0, 2),
    "cubic": (0,),
}


def free_lattice_indices(number) -> Tuple[int, ...]:
    return _FREE_LATTICE[crystal_system(number)]


def full_lattice(number, free: List[float]) -> Tuple[float, ...]:
    """Rebuild (a, b, c, alpha, beta, gamma) from the system's free parameters."""
    system = crystal_system(number)
    v = dict(zip(_FREE_LATTICE[system], free))
    if system == "triclinic":
        return tuple(v[i] for i in range(6))
    if system == "monoclinic":
        return (v[0], v[1], v[2], 90.0, v[4], 90.0)
    if system == "orthorhombic":
        return (v[0], v[1], v[2], 90.0, 90.0, 90.0)
    if system == "tetragonal":
        return (v[0], v[0], v[2], 90.0, 90.0, 90.0)
    if system == "hexagonal":
        return (v[0], v[0], v[2], 90.0, 90.0, 120.0)
    return (v[0], v[0], v[0], 90.0, 90.0, 90.0)


def _wrap_close(p, q, tol=1e-4) -> bool:
    return bool(np.all(np.abs(((p - q) + 0.5) % 1.0 - 0.5) < tol))


def _generator_point(wp, pos) -> np.ndarray:
    """The member of pos's orbit that lies on the position's own expression.

    Free parameters are read straight off the representative, which is only
    valid if the representative has the form the first operation describes --
    (x, x, x) for 32f in Fm-3m, not (-x, x, -x). Any orbit member fixed by that
    operation has the right form, so search the orbit for one.
    """
    op0 = wp.ops[0]
    for p in np.mod(np.array(wp.apply_ops(pos), dtype=float), 1.0):
        if _wrap_close(np.mod(op0.operate(p), 1.0), p):
            return p
    raise ValueError("no orbit member of %s lies on its generator form" % wp.get_label())


def encode(structure: Structure, symprec: float = SYMPREC) -> WyckoffCrystal:
    """Structure -> Wyckoff form, in the standard setting of its space group.

    Oxidation states are dropped: they are assigned separately per orbit, and
    left on the species they would make symmetry-equivalent sites that differ
    only in charge into different orbits.
    """
    from pyxtal import pyxtal

    s = structure.copy()
    s.remove_oxidation_states()
    xtal = pyxtal()
    xtal.from_seed(s, tol=symprec)

    sites = []
    for site in xtal.atom_sites:
        wp = site.wp
        free_idx = _free_indices(wp)
        rep = _generator_point(wp, np.asarray(site.position, dtype=float))
        sites.append(WyckoffSite(
            element=str(site.specie),
            wp_index=int(wp.index),
            label=wp.get_label(),
            multiplicity=int(wp.multiplicity),
            free=[float(rep[j]) for j in free_idx],
        ))

    a, b, c, alpha, beta, gamma = xtal.lattice.get_para(degree=True)
    return WyckoffCrystal(
        space_group=int(xtal.group.number),
        lattice=(a, b, c, alpha, beta, gamma),
        sites=sites,
    )


def decode(wc: WyckoffCrystal, merge_tol: float = 1e-6, return_orbits: bool = False):
    """Wyckoff form -> Structure, by expanding every orbit under the group.

    With return_orbits, also returns the orbit index of every atom, which is how
    per-orbit properties (an oxidation state) are carried onto the atoms.
    """
    group = _group(wc.space_group)
    lattice = Lattice.from_parameters(*wc.lattice)

    species, coords, orbit_of = [], [], []
    for k, site in enumerate(wc.sites):
        wp = group[site.wp_index]
        free_idx = _free_indices(wp)
        if len(site.free) != len(free_idx):
            raise ValueError("site %s carries %d free parameters, position %s has %d"
                             % (site.element, len(site.free), wp.get_label(), len(free_idx)))
        rep = np.zeros(3)
        for j, v in zip(free_idx, site.free):
            rep[j] = v
        # Applying the first operation pins the fixed coordinates to their exact
        # values (the 1/4 in "x,1/4,z") whatever rep carried for them.
        rep = wp.ops[0].operate(rep)
        orbit = np.mod(np.array(wp.apply_ops(rep), dtype=float), 1.0)

        # Only exact duplicates are merged. A looser tolerance would collapse an
        # orbit whose free parameter was predicted one bin from a special
        # position -- bin 0's centre is 0.00049, so x and -x sit 0.001 apart --
        # and the structure would silently lose atoms.
        kept = []
        for p in orbit:
            if not any(_wrap_close(p, q, merge_tol) for q in kept):
                kept.append(p)
        species.extend([site.element] * len(kept))
        coords.extend(kept)
        orbit_of.extend([k] * len(kept))

    structure = Structure(lattice, species, coords)
    return (structure, orbit_of) if return_orbits else structure


def to_primitive(structure: Structure, number: int, tol: float = 1e-4) -> Structure:
    """The primitive cell of a decoded conventional cell.

    decode() writes the conventional standard cell, which holds centering(number)
    lattice points, so a centred group's cell has 2-4 times the formula units of
    the primitive cell the benchmark formula describes. The centering vectors are
    exact in the decoded coordinates, so this is an exact fold, not a symmetry
    search: it never finds a smaller cell than the group's own.
    """
    letter = _group(number).symbol[0]
    if letter == "P":
        return structure
    basis = np.array(_PRIMITIVE_BASIS[letter], dtype=float)
    lattice = Lattice(basis @ structure.lattice.matrix)
    frac = np.mod(structure.frac_coords @ np.linalg.inv(basis), 1.0)
    keep = []
    for i, p in enumerate(frac):
        if not any(_wrap_close(p, frac[j], tol) for j in keep):
            keep.append(i)
    if len(keep) * _CENTERING[letter] != len(structure):
        raise ValueError("centering fold of group %d kept %d of %d atoms"
                         % (number, len(keep), len(structure)))
    return Structure(lattice, [structure[i].species for i in keep], frac[keep],
                     site_properties={k: [v[i] for i in keep] for k, v in structure.site_properties.items()})
