"""Analytic Gaussian and orientation integrals (the paper, App. E)."""

from fractions import Fraction

import numpy as np
import pytest

from bfss.algebra import E1, E2, E3
from bfss.gaussian import WISHART, OrientationIntegrals
from bfss.model import NU
from bfss.scalars import RATIONALS, Reals
from quadrature import SliceQuadrature


def evaluate(p, e):
    return sum(float(c) * e[0] ** m[0] * e[1] ** m[1] * e[2] ** m[2] for m, c in p.terms.items())


def test_low_wishart_moments_have_closed_forms():
    """<s> = 9/2 tr K^-1, <u> = 18 e2(K^-1), <v> = 63 det K^-1 and Var(s) = 9/2 tr K^-2."""
    assert WISHART.E0(1, 0, 0) == Fraction(9, 2) * E1
    assert WISHART.E0(0, 1, 0) == 18 * E2
    assert WISHART.E0(0, 0, 1) == 63 * E3
    assert WISHART.E0(2, 0, 0) == Fraction(99, 4) * E1**2 - 9 * E2


def test_wishart_moments_against_monte_carlo():
    rng = np.random.default_rng(7)
    K = np.array([[1.2, 0.3, -0.1], [0.3, 0.9, 0.2], [-0.1, 0.2, 1.5]])
    C = np.linalg.inv(K)
    e = (np.trace(C), (np.trace(C) ** 2 - np.trace(C @ C)) / 2, np.linalg.det(C))
    X = np.einsum("ab,nbi->nai", np.linalg.cholesky(C / 2), rng.standard_normal((400_000, 3, 9)))
    G = X @ X.transpose(0, 2, 1)
    s = np.trace(G, axis1=1, axis2=2)
    u = (s**2 - np.trace(G @ G, axis1=1, axis2=2)) / 2
    v = np.linalg.det(G)
    Cs = [np.eye(3), C, C @ C]

    def check(samples, exact):
        mean, err = samples.mean(0), samples.std(0) / np.sqrt(len(samples))
        assert np.all(np.abs(mean - exact) < 5 * err + 1e-12), (mean, exact, err)

    check(s * u, evaluate(WISHART.E0(1, 1, 0), e))
    check(s * v, evaluate(WISHART.E0(1, 0, 1), e))
    check(u * u, evaluate(WISHART.E0(0, 2, 0), e))
    for power, sigma, weight in [(1, (0, 1, 0), u), (2, (1, 0, 0), s), (1, (0, 0, 1), v)]:
        mat = sum(evaluate(p, e) * Cs[i] for i, p in enumerate(WISHART.moment(sigma, power)))
        Gk = G if power == 1 else G @ G
        check(weight[:, None, None] * Gk, mat)


@pytest.mark.parametrize("widths", [(2, 1, 2, 1), (3, 1, 4, 1), (4, 1, 2, 1)])
def test_boxed_orientation_formula(widths):
    """Delta_1^{1/2} int_0^1 z^{2r} Delta(z)^{-9/2-L} dz (the paper's Eq. (E.15)) against direct quadrature."""
    O = OrientationIntegrals(*widths)
    z, w = np.polynomial.legendre.leggauss(200)
    z, w = (z + 1) / 2, w / 2
    D = float(O.D0) - float(O.eta_jh) * z**2
    for r, L in [(0, 0), (1, 0), (2, 3), (3, 1), (0, 7)]:
        direct = np.sum(w * z ** (2 * r) * D ** (-4.5 - L)) * np.sqrt(float(O.D1))
        assert float(O.R(r, L)) == pytest.approx(direct, rel=1e-13)


@pytest.fixture(scope="module")
def grid():
    return SliceQuadrature(n=44)


def eigenvalue_quadrature(Q, a, b, c, d, sigma, atom):
    """Reduced int d^27X f_{a,b} f_{c,d} sigma(G) atom over singular values (independent oracle)."""
    (fj, Pj), (fh, Ph) = Q.moments(a, b), Q.moments(c, d)
    g, e = Q.g, Q.e
    poly = e[0] ** sigma[0] * e[1] ** sigma[1] * e[2] ** sigma[2]
    angular = {"1": fj * fh, "n1": (g * Pj).sum(0) * fh, "n2": (g**2 * Pj).sum(0) * fh,
               "m1": fj * (g * Ph).sum(0), "m2": fj * (g**2 * Ph).sum(0), "z1": (g * Pj * Ph).sum(0)}[atom]
    return Q.integrate(poly * angular, (b + d) / 2)


@pytest.mark.slow
@pytest.mark.parametrize("widths,sigma,atom", [
    ((2, 1, 2, 1), (0, 0, 0), "1"), ((2, 1, 3, 1), (1, 1, 0), "1"), ((3, 1, 4, 1), (0, 0, 1), "n1"),
    ((4, 1, 2, 1), (1, 0, 0), "m2"), ((2, 1, 3, 1), (0, 1, 0), "z1"), ((3, 2, 2, 1), (2, 0, 0), "n2"),
])
def test_master_integrals_against_eigenvalue_quadrature(grid, widths, sigma, atom):
    exact = float(OrientationIntegrals(*widths).master(sigma, atom))
    assert eigenvalue_quadrature(grid, *widths, sigma, atom) == pytest.approx(exact, rel=2e-7)


def test_masters_are_homogeneous_under_dilation():
    """X -> X / sqrt(lambda): reduced masters scale as lambda^{1/2 - nu - deg sigma - deg atom}."""
    lam = Fraction(3, 2)
    base, scaled = OrientationIntegrals(2, 1, 3, 1), OrientationIntegrals(2 * lam, lam, 3 * lam, lam)
    for sigma in [(0, 0, 0), (2, 1, 0), (1, 0, 1)]:
        for atom, extra in [("1", 0), ("n1", 1), ("m2", 2), ("z1", 1)]:
            degree = sigma[0] + 2 * sigma[1] + 3 * sigma[2] + extra
            e = Fraction(1, 2) - Fraction(NU) - degree
            assert scaled.master(sigma, atom) == base.master(sigma, atom) * RATIONALS(lam) ** int(e)


def test_ball_masters_enclose_the_exact_masters():
    """The same formulas in 128-bit ball arithmetic contain the exact rationals."""
    exact, balls = OrientationIntegrals(2, 1, 4, 1), OrientationIntegrals(2, 1, 4, 1, Reals(128))
    with balls.F.precision():
        for sigma in [(0, 0, 0), (3, 2, 1), (9, 3, 2)]:
            for atom in ("1", "n2", "m1", "z1"):
                ball = balls.master(sigma, atom)
                assert ball.contains(exact.master(sigma, atom)) and ball.rad() < 2**-100 * abs(ball.mid())
