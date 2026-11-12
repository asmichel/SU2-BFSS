"""Analytic Gaussian and orientation integrals (the paper, App. E).

For orientations n, m the product of two oriented profiles is exp(-tr(K G)) with
K = gamma I - eta_j n n^T - eta_h m m^T.  The columns of X are then
independent Gaussian vectors with covariance K^-1/2, and

  * ``WishartTable`` gives <sigma(G) G^power>_{X,K} (power = 0, 1, 2) for invariant monomials
    sigma = s^k u^l v^m as polynomials in the invariants e_1, e_2, e_3 of K^-1 times
    I, K^-1, K^-2, using Stein's identity <X_ai F> = ((K^-1)_ab / 2) <d_bi F>;
  * ``OrientationIntegrals`` integrates the result over the two orientations with
    the finite sum of the paper's Eq. (E.15) for int_0^1 z^{2r} Delta(z)^{-9/2-L} dz.

All results are rational functions of the widths, up to the common factor
pi^nu beta^{-1/2}, beta = (b_j + b_h)/2, which is divided out ("reduced" values); they are
evaluated exactly or in ball arithmetic (``scalars``).
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from math import comb

from .algebra import GENERATOR as Kinv  # K^-1
from .algebra import MatrixPolynomial, Polynomial
from .scalars import RATIONALS, Scalars


class WishartTable:
    """Exact Gaussian moments <s^k u^l v^m G^power>_{X,K}, power = 0, 1, 2, as functions of K^-1.

    E0 is a polynomial in the invariants (e1, e2, e3) of K^-1; E1 and E2 are matrix
    polynomials over (I, K^-1, K^-2).
    """

    @cache
    def E0(self, k, l, m) -> Polynomial:
        if (k, l, m) == (0, 0, 0):
            return Polynomial.const(1)
        if k:
            return self.E1(k - 1, l, m).trace()
        if l:
            return (self.E0(2, l - 1, m) - self.E2(0, l - 1, m).trace()) / 2
        adj = self.E2(0, 0, m - 1) - self.E1(1, 0, m - 1) + self.E0(0, 1, m - 1)
        return (Kinv * adj).trace() * Fraction(2 * m + 5, 6)

    @cache
    def E1(self, k, l, m) -> MatrixPolynomial:
        inner = MatrixPolynomial.const((9 + 2 * m) * self.E0(k, l, m))
        if k:
            inner += 2 * k * self.E1(k - 1, l, m)
        if l:
            inner += 2 * l * (self.E1(k + 1, l - 1, m) - self.E2(k, l - 1, m))
        return Kinv * inner / 2

    @cache
    def E2(self, k, l, m) -> MatrixPolynomial:
        inner = self.E0(k + 1, l, m) + (10 + 2 * l + 2 * m) * self.E1(k, l, m)
        if k:
            inner += 2 * k * self.E2(k - 1, l, m)
        if l:
            inner -= 2 * l * self.E0(k, l - 1, m + 1)
        return Kinv * inner / 2

    def moment(self, sigma, power) -> MatrixPolynomial:
        """<sigma G^power>_{X,K} over (I, K^-1, K^-2)."""
        return MatrixPolynomial.const(self.E0(*sigma)) if power == 0 else (self.E1, self.E2)[power - 1](*sigma)


WISHART = WishartTable()

# Orientation atoms multiplying invariant polynomials in the integrands:
#   "1", "n1" = n^T G n, "n2" = n^T G^2 n, "m1", "m2", "z1" = (n.m) n^T G m.
ATOMS = ("1", "n1", "n2", "m1", "m2", "z1")


class OrientationIntegrals:
    """Reduced orientation-averaged Gaussian integrals for one pair of profile widths.

    The orientation data are rational functions of the widths, formed exactly and only then
    taken into the number system (exact rationals or real balls): a ball around zero would
    turn every power of it into NaN.
    """

    def __init__(self, a, b, c, d, scalars: Scalars = RATIONALS):
        self.F = F = scalars
        a, b, c, d = map(Fraction, (a, b, c, d))
        gamma, eta_j, eta_h = (a + c) / 2, (a - b) / 2, (c - d) / 2
        D0 = (gamma - eta_j) * (gamma - eta_h)
        assert D0 > 0 and D0 > eta_j * eta_h
        unit = {0: {0: 1}}
        numerators = {  # orientation factors x^T K^-k y as {power of Delta: {power of z^2: coefficient}}
            ("1", 0): unit, ("n", 0): unit, ("m", 0): unit,
            ("n", 1): {1: {0: gamma - eta_h, 1: eta_h}},
            ("n", 2): {2: {0: (gamma - eta_h) ** 2, 1: eta_h * (2 * gamma - eta_h)}},
            ("m", 1): {1: {0: gamma - eta_j, 1: eta_j}},
            ("m", 2): {2: {0: (gamma - eta_j) ** 2, 1: eta_j * (2 * gamma - eta_j)}},
            ("z", 0): {0: {1: 1}}, ("z", 1): {1: {1: gamma}},
            ("z", 2): {2: {1: gamma**2 - eta_j * eta_h, 2: eta_j * eta_h}},
        }
        with F.precision():
            self.gamma, self.eta_jh, self.D0, self.D1 = F(gamma), F(eta_j * eta_h), F(D0), F(D0 - eta_j * eta_h)
            self.k1, self.k2 = F(3 * gamma - eta_j - eta_h), F(2 * gamma - eta_j - eta_h)
            self.numerators = {key: {e: {r: F(x) for r, x in z.items()} for e, z in v.items()}
                               for key, v in numerators.items()}
        self.R, self.emonomial, self.master = cache(self.R), cache(self.emonomial), cache(self.master)

    def R(self, r, L):
        """Delta_1^{1/2} int_0^1 z^{2r} Delta(z)^{-9/2-L} dz (rational), the finite sum of the paper's Eq. (E.15)."""
        n = 3 + L - r
        assert n >= 0
        return self.D0 ** (r - 4 - L) * sum(comb(n, t) * self.eta_jh**t / ((2 * r + 2 * t + 1) * self.D1 ** (r + t))
                                            for t in range(n + 1))

    def emonomial(self, i, j, k, factor):
        """Reduced integral of pi^nu (det K)^{-9/2} e1^i e2^j e3^k times an orientation factor, e1, e2, e3 the
        invariants of K^-1.

        With det K = gamma Delta: e1 = (gamma k2 + Delta)/(gamma Delta), e2 = k1/(gamma Delta), e3 = 1/(gamma Delta),
        k1 = 3 gamma - eta_j - eta_h, k2 = 2 gamma - eta_j - eta_h; gamma^{-9/2} Delta_1^{-1/2} = gamma^{-5} beta^{-1/2}.
        """
        gamma, k1, k2 = self.gamma, self.k1, self.k2
        total = self.F(0)
        n = i + j + k
        for t in range(i + 1):
            pref = comb(i, t) * (gamma * k2) ** (i - t) * k1**j
            for dpow, zpoly in self.numerators[factor].items():
                for r, c in zpoly.items():
                    total += pref * c * self.R(r, n - t + dpow)
        return total / gamma ** (5 + n)

    def contract(self, poly: Polynomial, factor):
        F = self.F
        return sum((F(c) * self.emonomial(*m, factor) for m, c in poly.terms.items()), F(0))

    def master(self, sigma, atom):
        """Reduced int d^27X f_j f_h sigma(G) atom, sigma = (k, l, m) exponents of s, u, v."""
        kind, power = atom[0], int(atom[1:] or 0)
        with self.F.precision():
            moment = WISHART.moment(sigma, power)
            return sum((self.contract(p, (kind, j)) for j, p in enumerate(moment)), self.F(0))
