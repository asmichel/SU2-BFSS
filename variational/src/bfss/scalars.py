"""Number systems for matrix elements: exact rationals or real balls of tunable precision.

Every matrix element is a rational expression in rational data (contraction coefficients,
profile widths, Gaussian moments), so it can be evaluated in any field containing Q.
``Rationals`` keeps it exact (FLINT ``fmpq``); ``Reals(prec)`` evaluates it in ball
arithmetic with ``prec`` bits (FLINT ``arb``), which is much faster for large bases and
still carries a rigorous error bound.  Both provide scalars and dense matrices with the
same interface, so the assembly is written once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext

import flint
from flint import arb, arb_mat, fmpq, fmpq_mat, fmpz_mat

from .algebra import rational


def ball(q) -> arb:
    """Ball enclosing a rational number at the current working precision."""
    q = rational(q)
    return arb(q.numerator) / q.denominator


class Scalars(ABC):
    """A field containing Q in which matrix elements are evaluated."""

    exact: bool

    @abstractmethod
    def __call__(self, q):
        """The element representing a rational number (int, Fraction or decimal string)."""

    @abstractmethod
    def matrix(self, rows):
        """Dense matrix from nested lists of elements."""

    @abstractmethod
    def integer_matrix(self, rows, denominator: int):
        """Dense matrix of nested lists of integers divided by a common denominator."""

    @abstractmethod
    def sqrt(self, x):
        """Square root of a positive element (or rational number)."""

    def precision(self):
        """Context in which arithmetic on the elements takes place."""
        return nullcontext()


class Rationals(Scalars):
    exact = True

    def __call__(self, q):
        if isinstance(q, fmpq):
            return q
        q = rational(q)
        return fmpq(q.numerator, q.denominator)

    def matrix(self, rows):
        return fmpq_mat(rows)

    def integer_matrix(self, rows, denominator):
        return fmpq_mat(fmpz_mat(rows)) / denominator

    def sqrt(self, x):
        return self(x).sqrt()  # raises unless x is the square of a rational

    def __repr__(self):
        return "Rationals()"


class Reals(Scalars):
    """Real balls with ``prec`` bits of working precision."""

    exact = False

    def __init__(self, prec: int = 256):
        self.prec = prec

    def __call__(self, q):
        return ball(q)

    def matrix(self, rows):
        return arb_mat(rows)

    def integer_matrix(self, rows, denominator):
        return arb_mat(fmpz_mat(rows)) / denominator

    def sqrt(self, x):
        return (x if isinstance(x, arb) else ball(x)).sqrt()

    def precision(self):
        return flint.ctx.workprec(self.prec)

    def __repr__(self):
        return f"Reals({self.prec})"


RATIONALS = Rationals()
