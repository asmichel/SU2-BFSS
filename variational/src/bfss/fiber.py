"""Flavour-orbit representation of the stabilizer-invariant fermion space.

At a point D(x) of the gauge slice every polynomial singlet takes values in the invariant
subspace F_* of the stabilizer K_4 x SO(6) (the paper, Sec. III A and App. D).  Spin(6) ~ SU(4) acts on the
flavour index, and SU(4) invariants are balanced (k fermions per flavour) and
symmetric under flavour permutations with the plain (unsigned) orbit convention of
the JSON data.  We therefore work in the span of flavour-orbit vectors

    [q] = sum over distinct permutations of the four 6-bit flavour masks of q,

graded by the characters of K_4, the simultaneous colour/space
half-turns.  The half-turn about axis 1 reverses the six orbitals of each flavour,
the half-turn about axis 3 is diagonal (the paper's Eq. (D.5)); both are computed from the
paper's conventions (App. A) in ``klein_signature``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations_with_replacement
from math import factorial, lcm

import numpy as np
import scipy.sparse as sp

from .fock import Q8, FermionOperator, apply_word, dagger

UP = 0b101010  # orbitals o = 2c + rho with rho = 1
NONZERO_WEIGHT = 0b110011  # orbitals with spherical colour weight -1 or +1 (c = 0, 2)
REVERSE = np.array([int(f"{o:06b}"[::-1], 2) for o in range(64)])

# K_4 characters (eigenvalue of the half-turn about axis 1, about axis 3); the
# derivative across the colour/space plane {b, k} lands in the character of the
# remaining axis, labelled here by that plane.
TRIVIAL = (1, 1)
PLANE_CHARACTER = {(0, 1): (-1, 1), (0, 2): (-1, -1), (1, 2): (1, -1)}


def flavor_fields(masks) -> np.ndarray:
    """The four 6-bit orbital masks (one per flavour) of Fock occupation masks, along a new last axis."""
    return np.stack([(np.asarray(masks) >> (6 * A)) & 63 for A in range(4)], axis=-1)


def orbit_key(fields) -> np.ndarray:
    """Key of the flavour orbit of the given fields: sorted fields packed into one mask."""
    f = np.sort(np.asarray(fields, dtype=np.int64), axis=-1)
    return f[..., 0] | f[..., 1] << 6 | f[..., 2] << 12 | f[..., 3] << 18


def canonical(masks: np.ndarray) -> np.ndarray:
    """Orbit keys of Fock masks; equal keys label one flavour orbit."""
    return orbit_key(flavor_fields(masks))


def klein_signature(keys: np.ndarray):
    """(orbit key under the axis-1 half-turn, sign of the axis-3 half-turn) on balanced orbits."""
    f = flavor_fields(keys)
    odd = (np.bitwise_count(f & UP).astype(np.int64) + np.bitwise_count(f & NONZERO_WEIGHT)).sum(-1)
    return orbit_key(REVERSE[f]), 1 - 2 * (odd & 1)


class OrbitSpace:
    """Balanced flavour orbits; ``keys`` double as the representative Fock masks."""

    def __init__(self):
        keys, sizes = [], []
        for k in range(7):
            masks = [m for m in range(64) if m.bit_count() == k]
            for q in combinations_with_replacement(masks, 4):
                keys.append(int(orbit_key(q)))
                sizes.append(factorial(4) // np.prod([factorial(n) for n in Counter(q).values()]))
        order = np.argsort(keys)
        self.keys = np.array(keys, dtype=np.int64)[order]
        self.sizes = np.array(sizes, dtype=np.int64)[order]

    def __len__(self):
        return len(self.keys)

    def index(self, masks: np.ndarray):
        """Orbit index of arbitrary Fock masks and whether they are balanced orbit members."""
        key = canonical(masks)
        idx = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        return idx, self.keys[idx] == key

    @cached_property
    def sectors(self) -> dict:
        """K_4 sectors: representative orbits, partner orbits and character signs."""
        image, phase = klein_signature(self.keys)
        partner, ok = self.index(image)
        assert ok.all()
        out = {}
        for e1 in (1, -1):
            for e3 in (1, -1):
                rep = np.nonzero((phase == e3) & (np.arange(len(self)) <= partner) & ((partner != np.arange(len(self))) | (e1 == 1)))[0]
                out[(e1, e3)] = Sector(rep, partner[rep], e1, self.sizes[rep] * np.where(partner[rep] == rep, 1, 2))
        return out


@dataclass
class Sector:
    """Basis [q] + e1 [R1 q] of one K_4 sector, with inner-product weights."""

    rep: np.ndarray
    partner: np.ndarray
    e1: int
    weight: np.ndarray

    def __len__(self):
        return len(self.rep)

    def embed_full(self, n_orbits: int) -> sp.csr_matrix:
        """Orbit-space columns [q] + e1 [R1 q] of the sector basis, shape (n_orbits, dim)."""
        paired = self.partner != self.rep
        rows = np.concatenate([self.rep, self.partner[paired]])
        cols = np.concatenate([np.arange(len(self)), np.nonzero(paired)[0]])
        vals = np.concatenate([np.ones(len(self), dtype=np.int64), np.full(paired.sum(), self.e1, dtype=np.int64)])
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_orbits, len(self)))


class OrbitOperator:
    """Exact matrix of an operator that maps orbit space into itself (rows read at the representative masks).

    Entries lie in Q(zeta_8); they are stored as four integer COO arrays (the
    coordinates on 1, z, z^2, z^3) over a common denominator.
    """

    def __init__(self, space: OrbitSpace, op: FermionOperator):
        self.space = space
        den = lcm(1, *(x.denominator for c in op.terms.values() for x in c.c))
        rows, cols, vals = [], [], []
        reps = np.arange(len(space))
        for word, c in op.terms.items():
            image, sign, valid = apply_word(tuple(dagger(l) for l in reversed(word)), space.keys)
            idx, member = space.index(image)
            keep = valid & member
            if keep.any():
                coef = np.array([int(x * den) for x in c.c], dtype=np.int64)
                rows.append(reps[keep])
                cols.append(idx[keep])
                vals.append(np.outer(sign[keep], coef))
        self.den = den
        grades = {c.grade for c in op.terms.values()}
        assert len(grades) <= 1, "operator mixes real and imaginary orbit-basis entries"
        self.grade = grades.pop() if grades else 0
        if rows:
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            vals = np.concatenate(vals)
            key, inverse = np.unique(rows * len(space) + cols, return_inverse=True)
            summed = np.zeros((len(key), 4), dtype=np.int64)
            np.add.at(summed, inverse, vals)
            keep = summed.any(axis=1)
            assert np.abs(summed).max() < 128
            self.rows = (key[keep] // len(space)).astype(np.int32)
            self.cols = (key[keep] % len(space)).astype(np.int32)
            self.vals = summed[keep].astype(np.int8)
        else:
            self.rows = self.cols = np.zeros(0, dtype=np.int64)
            self.vals = np.zeros((0, 4), dtype=np.int64)

    def mod(self, p: int, zeta: int) -> sp.csr_matrix:
        """Image in F_p (z -> zeta) as a sparse int64 matrix with entries in [0, p)."""
        w = [pow(zeta, k, p) for k in range(4)]
        vals = self.vals.astype(np.int64)
        v = sum(vals[:, k] % p * w[k] % p for k in range(4)) % p * pow(self.den, -1, p) % p
        n = len(self.space)
        return reduce_mod(sp.csr_matrix((v.astype(np.int64), (self.rows, self.cols)), shape=(n, n)), p)

    def to_complex(self) -> sp.csr_matrix:
        """Image in C (z -> exp(i pi/4)) as a sparse complex matrix."""
        z = np.exp(1j * np.pi / 4) ** np.arange(4)
        n = len(self.space)
        return sp.csr_matrix((self.vals.astype(float) @ z / self.den, (self.rows, self.cols)), shape=(n, n))


def reduce_mod(a, p: int):
    """Entries reduced into [0, p): arrays elementwise, sparse matrices as CSR without explicit zeros."""
    if not sp.issparse(a):
        return a % p
    a = a.tocsr()
    a.data %= p
    a.eliminate_zeros()
    return a


class SliceArithmetic(ABC):
    """Realization of the slice operators and seed data over a field: exact images in F_p or complex numbers.

    This is the interface the covariant expression evaluator acts through.  Its ``sectors`` are the K_4 sectors of
    orbit space (3967 dimensions in the trivial one), in which the operators are sparse.
    """

    dtype = np.int64

    def __init__(self, operators: dict, space: OrbitSpace):
        self.operators, self.space, self._full, self._restricted = operators, space, {}, {}
        self.sectors = space.sectors
        self.embed = {ch: s.embed_full(len(space)) for ch, s in self.sectors.items()}

    @abstractmethod
    def scalar(self, q: Q8):
        """Image of an exact scalar."""

    @abstractmethod
    def image(self, op: OrbitOperator) -> sp.csr_matrix:
        """Image of an exact orbit-space operator."""

    def reduce(self, a):
        return a

    def grade(self, name):
        return self.operators[name].grade

    def full(self, name):
        if name not in self._full:
            self._full[name] = self.image(self.operators[name])
        return self._full[name]

    def act(self, name, v):
        return self.reduce(self.full(name) @ self.reduce(v))

    def orbit_vector(self, rows, denominator):
        """Orbit coordinates of published tensor rows [sorted flavour masks, scalar]."""
        v = np.zeros(len(self.space), dtype=self.dtype)
        keys = orbit_key([q for q, _ in rows])
        idx, ok = self.space.index(keys)
        assert ok.all()
        v[idx] = [self.scalar(Q8.from_json(c) / denominator) for _, c in rows]
        return v

    def inner(self, u, v):
        """<u, v> = sum over orbits of size * conj(u) v (complex realization)."""
        return complex(np.vdot(u, self.space.sizes * v))

    # sectors
    def restricted(self, name, src=TRIVIAL, dst=TRIVIAL):
        """Matrix of an operator from K_4 sector src to sector dst (sparse)."""
        key = (name, src, dst)
        if key not in self._restricted:
            self._restricted[key] = self.reduce(self.full(name)[self.sectors[dst].rep] @ self.embed[src])
        return self._restricted[key]


class ComplexContext(SliceArithmetic):
    dtype = complex

    def scalar(self, q):
        return complex(q)

    def image(self, op):
        return op.to_complex()
