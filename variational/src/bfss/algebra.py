"""Linear combinations over a basis, and the polynomial algebras of the construction.

``LinearCombination`` is an element of a free module: finitely many basis keys with
nonzero coefficients in a ring.  Scalars (whatever the coefficient ring accepts) act on
the coefficients; an algebra defines the product of two elements.

  * ``Polynomial``: monomials in three variables (rational coefficients here, Q(zeta_8)
    for the gauge-slice polynomials of ``covariants``);
  * ``MatrixPolynomial``: a I + b A + c A^2 with coefficients polynomial in the
    invariants (e1, e2, e3) of a symmetric 3x3 matrix A -- (s, u, v) for G = X X^T, or
    those of the covariance K^-1 -- reduced with Cayley-Hamilton, A^3 = e1 A^2 - e2 A + e3 I;
  * ``fock.FermionOperator``: normal-ordered words of fermionic ladder operators.
"""

from __future__ import annotations

from fractions import Fraction


def rational(c) -> Fraction:
    return c if type(c) is Fraction else Fraction(c)


class LinearCombination:
    """Element of a free module: basis key -> nonzero coefficient."""

    coerce = staticmethod(rational)  # into the coefficient ring
    unit = ()  # basis key of the multiplicative unit

    def __init__(self, terms=None):
        self.terms = {}
        for key, c in (terms or {}).items():
            self._add(key, self.coerce(c))

    def _add(self, key, c):
        if c:
            s = self.terms[key] + c if key in self.terms else c
            if s:
                self.terms[key] = s
            else:
                del self.terms[key]

    @classmethod
    def const(cls, c):
        return cls({cls.unit: c})

    def _lift(self, o):
        """o as an element of this module (a scalar as a multiple of the unit), or None."""
        if isinstance(o, type(self)):
            return o
        try:
            return self.const(o)
        except TypeError:
            return None

    def __add__(self, o):
        o = self._lift(o)
        if o is None:
            return NotImplemented
        out = type(self)(self.terms)
        for key, c in o.terms.items():
            out._add(key, c)
        return out

    __radd__ = __add__

    def __neg__(self):
        return type(self)({key: -c for key, c in self.terms.items()})

    def __sub__(self, o):
        return self + -o

    def __rsub__(self, o):
        return -self + o

    def __mul__(self, o):
        if isinstance(o, type(self)):
            return self.multiply(o)
        try:
            o = self.coerce(o)
        except TypeError:
            return NotImplemented
        return type(self)({key: c * o for key, c in self.terms.items()})

    def __rmul__(self, o):
        return self * o  # only scalars arrive here, and they commute

    def __truediv__(self, o):
        return type(self)({key: c / o for key, c in self.terms.items()})

    def __pow__(self, n):
        out = self.const(1)
        for _ in range(n):
            out = out * self
        return out

    def multiply(self, o):
        """Product of two elements (algebras only)."""
        raise TypeError(f"{type(self).__name__} has no product")

    def __eq__(self, o):
        o = self._lift(o)
        return o is not None and self.terms == o.terms

    def __bool__(self):
        return bool(self.terms)

    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        return f"{type(self).__name__}({self.terms})"


class Polynomial(LinearCombination):
    """Polynomial in three variables: exponent triple -> coefficient."""

    unit = (0, 0, 0)

    @classmethod
    def monomial(cls, exponents, c=1):
        """c times the monomial with these exponents (zero if one is negative)."""
        return cls({tuple(exponents): c} if min(exponents) >= 0 else {})

    @classmethod
    def variable(cls, i):
        return cls.monomial([int(j == i) for j in range(3)])

    def derivative(self, i):
        return type(self)({m[:i] + (m[i] - 1,) + m[i + 1:]: c * m[i] for m, c in self.terms.items() if m[i]})

    def multiply(self, o):
        out = type(self)()
        for m1, c1 in self.terms.items():
            for m2, c2 in o.terms.items():
                out._add((m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]), c1 * c2)
        return out


E1, E2, E3 = (Polynomial.variable(i) for i in range(3))
ZERO = Polynomial()


def invariants(g):
    """Values of (E1, E2, E3) at the diagonal matrix diag(g), e.g. (s, u, v) at D(x) for g = x^2."""
    return g[0] + g[1] + g[2], g[0] * g[1] + g[0] * g[2] + g[1] * g[2], g[0] * g[1] * g[2]


class MatrixPolynomial(LinearCombination):
    """a I + b A + c A^2 (keys 0, 1, 2) with coefficients polynomial in the invariants E1, E2, E3 of A."""

    coerce = staticmethod(lambda c: c if isinstance(c, Polynomial) else Polynomial.const(c))
    unit = 0

    def __iter__(self):
        return (self.terms.get(k, ZERO) for k in range(3))

    def _times_generator(self):
        a, b, c = self
        return MatrixPolynomial({0: E3 * c, 1: a - E2 * c, 2: b + E1 * c})

    def multiply(self, o):
        a, b, c = self
        Ao = o._times_generator()
        return o * a + Ao * b + Ao._times_generator() * c

    def trace(self) -> Polynomial:
        a, b, c = self
        return 3 * a + E1 * b + (E1 * E1 - 2 * E2) * c


GENERATOR = MatrixPolynomial({1: 1})
