"""Exact fermionic operator algebra on the 24-mode Fock space.

Scalars live in the cyclotomic field Q(zeta_8), which contains i and sqrt(2) and
therefore every coefficient produced by the Majorana, gamma and color conventions
of the paper (App. A).  Operators are finite sums of normal-ordered words in the ladder
operators; a word is a tuple of letters ``2m`` (f_m) and ``2m+1`` (f_m^dagger),
multiplied left to right, so the rightmost letter acts first on a ket.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import reduce

import numpy as np

from .algebra import LinearCombination


class Q8:
    """Element a0 + a1 z + a2 z^2 + a3 z^3 of Q(z), z = exp(i pi/4), z^4 = -1."""

    __slots__ = ("c",)

    def __init__(self, c=(0, 0, 0, 0)):
        self.c = tuple(Fraction(x) for x in c)

    @staticmethod
    def coerce(x) -> Q8:
        return x if isinstance(x, Q8) else Q8((x, 0, 0, 0))

    @classmethod
    def from_json(cls, s) -> Q8:
        """Decode the JSON scalar [a, c, b, d] = a + c sqrt2 + i (b + d sqrt2)."""
        a, c, b, d = (Fraction(x) for x in s)
        return cls((a, c + d, b, d - c))

    def __add__(self, o):
        o = Q8.coerce(o)
        return Q8(tuple(x + y for x, y in zip(self.c, o.c)))

    __radd__ = __add__

    def __neg__(self):
        return Q8(tuple(-x for x in self.c))

    def __sub__(self, o):
        return self + (-Q8.coerce(o))

    def __rsub__(self, o):
        return Q8.coerce(o) - self

    def __mul__(self, o):
        o = Q8.coerce(o)
        r = [Fraction(0)] * 4
        for i, x in enumerate(self.c):
            if x:
                for j, y in enumerate(o.c):
                    if y:
                        k = i + j
                        r[k % 4] += x * y if k < 4 else -x * y
        return Q8(r)

    __rmul__ = __mul__

    def __truediv__(self, o):
        """Division by a rational number."""
        return Q8(tuple(x / Fraction(o) for x in self.c))

    def conjugate(self) -> Q8:
        a0, a1, a2, a3 = self.c
        return Q8((a0, -a3, -a2, -a1))

    def __bool__(self):
        return any(self.c)

    @property
    def grade(self) -> int:
        """0 for real elements (a + b sqrt2), 1 for imaginary ones; mixed elements are rejected."""
        a0, a1, a2, a3 = self.c
        if a2 == 0 and a3 == -a1:
            return 0
        if a0 == 0 and a3 == a1:
            return 1
        raise ValueError(f"{self} is neither real nor imaginary")

    def __eq__(self, o):
        return self.c == Q8.coerce(o).c

    def __hash__(self):
        return hash(self.c)

    def __complex__(self):
        z = np.exp(1j * np.pi / 4)
        return complex(sum(float(x) * z**k for k, x in enumerate(self.c)))

    def __repr__(self):
        return f"Q8{tuple(str(x) for x in self.c)}"

    def mod(self, p: int, zeta: int) -> int:
        """Image under Z[zeta_8, 1/2...] -> F_p sending z to the given 8th root of unity."""
        return sum(x.numerator * pow(x.denominator, -1, p) * pow(zeta, k, p) for k, x in enumerate(self.c)) % p


I = Q8((0, 0, 1, 0))
SQRT2 = Q8((0, 1, 0, -1))


def dagger(letter: int) -> int:
    return letter ^ 1


class FermionOperator(LinearCombination):
    """Finite linear combination of normal-ordered ladder-operator words (the empty word is 1)."""

    coerce = staticmethod(Q8.coerce)

    @classmethod
    def ladder(cls, mode: int, create: bool) -> FermionOperator:
        return cls({(2 * mode + int(create),): 1})

    @classmethod
    def from_words(cls, entries) -> FermionOperator:
        """Sum of (word, coefficient) pairs given in arbitrary operator order."""
        return reduce(lambda acc, wc: acc + cls._normal_order(*wc), entries, cls())

    @staticmethod
    def _key(letter):
        # creators (odd letters) first in ascending mode, annihilators after in descending mode
        return (0, letter) if letter & 1 else (1, -letter)

    @classmethod
    def _normal_order(cls, word, coef) -> FermionOperator:
        out, todo = cls(), [(tuple(word), Q8.coerce(coef))]
        while todo:
            w, c = todo.pop()
            for k in range(len(w) - 1):
                a, b = w[k], w[k + 1]
                if a == b:
                    break  # f f = f^dagger f^dagger = 0
                if cls._key(a) > cls._key(b):
                    swapped = w[:k] + (b, a) + w[k + 2 :]
                    todo.append((swapped, -c))
                    if a >> 1 == b >> 1:  # {f_m, f_m^dagger} = 1
                        todo.append((w[:k] + w[k + 2 :], c))
                    break
            else:
                out._add(w, c)
        return out

    def multiply(self, o):
        acc = defaultdict(Q8)
        for w1, c1 in self.terms.items():
            for w2, c2 in o.terms.items():
                acc[w1 + w2] = acc[w1 + w2] + c1 * c2
        out = FermionOperator()
        for w, c in acc.items():
            for w2, c2 in self._normal_order(w, c).terms.items():
                out._add(w2, c2)
        return out

    def adjoint(self) -> FermionOperator:
        return FermionOperator.from_words((tuple(dagger(l) for l in reversed(w)), c.conjugate()) for w, c in self.terms.items())

    def commutator(self, o) -> FermionOperator:
        return self * o - o * self

    def __repr__(self):
        return f"FermionOperator({len(self.terms)} terms)"


def apply_word(word, masks: np.ndarray):
    """Act with a word on occupation states |mask>; returns (new masks, signs, valid)."""
    masks = masks.astype(np.int64).copy()
    sign = np.ones(masks.shape, dtype=np.int64)
    valid = np.ones(masks.shape, dtype=bool)
    for letter in reversed(word):
        m, create = letter >> 1, letter & 1
        bit = np.int64(1) << m
        occupied = (masks & bit) != 0
        valid &= occupied != bool(create)
        below = np.bitwise_count(masks & (bit - 1)).astype(np.int64)
        sign *= 1 - 2 * (below & 1)
        masks ^= bit
    return masks, sign, valid
