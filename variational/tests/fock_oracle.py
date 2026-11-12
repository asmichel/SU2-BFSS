"""Brute-force evaluation in the full 2^24-dimensional Fock space (sparse vectors).

Used as an independent oracle: covariants and all 27 Cartesian derivatives are formed
directly from the Yukawa operators at a point D(x) of the gauge slice, without the orbit
representation, K_4 sectors, equivariance formulas or modular arithmetic.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from bfss.fock import FermionOperator, apply_word


class FockVector:
    def __init__(self, masks, amps):
        order = np.argsort(masks)
        self.masks, self.amps = np.asarray(masks, dtype=np.int64)[order], np.asarray(amps, dtype=complex)[order]

    @classmethod
    def combine(cls, masks, amps):
        if len(masks) == 0:
            return cls(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=complex))
        key, inv = np.unique(masks, return_inverse=True)
        out = np.zeros(len(key), dtype=complex)
        np.add.at(out, inv, amps)
        keep = np.abs(out) > 0
        return cls(key[keep], out[keep])

    def __add__(self, o):
        return FockVector.combine(np.concatenate([self.masks, o.masks]), np.concatenate([self.amps, o.amps]))

    def __rmul__(self, c):
        return FockVector(self.masks, self.amps * c)

    def inner(self, o) -> complex:
        common, i, j = np.intersect1d(self.masks, o.masks, assume_unique=True, return_indices=True)
        return complex(np.vdot(self.amps[i], o.amps[j]))


def apply(op: FermionOperator, v: FockVector) -> FockVector:
    ms, ams = [], []
    for word, c in op.terms.items():
        new, sign, valid = apply_word(word, v.masks)
        ms.append(new[valid])
        ams.append(complex(c) * sign[valid] * v.amps[valid])
    return FockVector.combine(np.concatenate(ms), np.concatenate(ams)) if ms else FockVector([], [])


def orbit_tensor(rows, denominator) -> FockVector:
    """Fock vector of published orbit rows: every distinct flavour permutation gets the row value."""
    from bfss.fock import Q8
    ms, ams = [], []
    for q, c in rows:
        value = complex(Q8.from_json(c)) / denominator
        for perm in set(permutations(q)):
            ms.append(sum(m << (6 * A) for A, m in enumerate(perm)))
            ams.append(value)
    return FockVector.combine(np.array(ms), np.array(ams))
