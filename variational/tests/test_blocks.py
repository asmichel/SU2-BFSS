"""Observable blocks between trial wavefunctions (bfss.blocks)."""

from fractions import Fraction

import numpy as np
import pytest

from bfss.blocks import BLOCKS, Assembler
from bfss.scalars import RATIONALS, Reals
from bfss.model import NU
from conftest import SMALL_CORES
from quadrature import SliceQuadrature

WIDTHS = ((Fraction(2), Fraction(1)), (Fraction(3), Fraction(1)))
SCALARS = ((0, 0, 0), (1, 0, 0), (0, 0, 1))


def trials(widths=WIDTHS, scalars=SCALARS):
    return [(a, b, e, c) for a, b in widths for e in scalars for c in range(len(SMALL_CORES))]


def degree(trial, table):
    k, l, m = trial[2]
    return 2 * k + 4 * l + 6 * m + table.degrees[trial[3]]


@pytest.fixture(scope="module")
def blocks(small_table, small_seeds):
    A = Assembler(trials(), small_table, small_seeds.norm)
    return {b: A.matrix(b) for b in BLOCKS}


def floats(M):
    return np.array([[float(x) for x in row] for row in M.tolist()])


def test_blocks_are_exactly_symmetric(blocks):
    """Exact symmetry includes the one-sided frakB and frakV forms, i.e. their Hermiticity."""
    for name, M in blocks.items():
        assert M == M.transpose(), name


def test_ball_blocks_enclose_the_exact_blocks(blocks, small_table, small_seeds):
    """The same assembly in 128-bit ball arithmetic rigorously encloses every exact entry, tightly."""
    balls = Assembler(trials(), small_table, small_seeds.norm, Reals(128))
    for name, exact in blocks.items():
        B = balls.matrix(name)
        assert B.contains(exact), name
        radius = max(float(x.rad()) for row in B.tolist() for x in row)
        assert radius < 2.0**-100 * np.abs(floats(exact)).max(), name


def test_positivity(blocks):
    """M > 0 and H = Q_alpha^2, H_aux = (1/16) sum_alpha Q_alpha R^2 Q_alpha, R^2 to R^8 are nonnegative on the span."""
    F = lambda name: floats(blocks[name])
    M = F("M")
    L = np.linalg.cholesky(M)
    Li = np.linalg.inv(L)
    for parts in (("T", "u", "Y"), ("Haux_T", "Haux_u", "Haux_Y"), ("R2",), ("R4",), ("R6",), ("R8",)):
        B = sum(F(p) for p in parts)
        assert np.linalg.eigvalsh(Li @ B @ Li.T).min() > 0


def test_dilation_laws(small_table, small_seeds):
    """B(lambda widths) = lambda^(1/2 - nu - (d_j + d_h)/2 + e_B) B(widths) for every block."""
    lam = Fraction(3, 2)
    base = Assembler(trials(), small_table, small_seeds.norm)
    scaled = Assembler(trials(tuple((a * lam, b * lam) for a, b in WIDTHS)), small_table, small_seeds.norm)
    T = trials()
    for name in BLOCKS:
        B0, B1 = base.matrix(name), scaled.matrix(name)
        for j in range(0, len(T), 5):
            for h in range(len(T)):
                if B0[j, h]:
                    e = Fraction(1, 2) - Fraction(NU) - Fraction(degree(T[j], small_table) + degree(T[h], small_table), 2) \
                        + BLOCKS[name].exponent
                    assert e.denominator == 1
                    assert B1[j, h] == B0[j, h] * RATIONALS(lam) ** int(e)


class SliceIntegrands:
    """Gauge-slice form of the block integrands, integrated over singular values."""

    def __init__(self, table, norm, widths=WIDTHS):
        self.Q, self.table, self.norm = SliceQuadrature(), table, norm
        self.e = self.Q.e
        self.profiles = {w: self.Q.profile(*map(float, w)) for w in widths}

    def family(self, f, j, h):
        ft = self.table.families[f]
        hit = ft.lookup().get((j, h))
        if hit is None:
            return 0.0
        N, row = hit
        mono = lambda m: self.e[0] ** m[0] * self.e[1] ** m[1] * self.e[2] ** m[2]
        c = [float(x) * float(self.norm) for x in ft.coefficients(N, row)]
        if ft.diagonal:
            return np.stack([sum(ci * self.Q.g[a] ** k * mono(m) for (k, m), ci in zip(ft.basis(N), c))
                             for a in range(3)])
        return sum(ci * mono(m) for m, ci in zip(ft.basis(N), c))

    def scalar(self, trial):
        """F = f s^k u^l v^m on the slice and its gradient along the slice."""
        f, df = self.profiles[trial[:2]]
        k, l, m = trial[2]
        s, u, v = self.e
        g = self.Q.g
        w = s**k * u**l * v**m
        # d w / d x_a = 2 x_a d w / d g_a with ds = 1, du = s - g_a, dv = v / g_a
        dg = [k * s ** max(k - 1, 0) * u**l * v**m + l * s**k * u ** max(l - 1, 0) * v**m * (s - g[a])
              + m * s**k * u**l * v ** max(m - 1, 0) * (v / g[a]) for a in range(3)]
        return f * w, df * w + f * np.stack([2 * self.Q.x[a] * dg[a] for a in range(3)])

    def element(self, name, tj, th):
        j, h = tj[3], th[3]
        Fj, dFj = self.scalar(tj)
        Fh, dFh = self.scalar(th)
        s, u, _ = self.e
        x = self.Q.x
        fam = lambda f, a=j, b=h: self.family(f, a, b)
        T = 0.5 * ((dFj * dFh).sum(0) * fam("g") + (dFj * Fh * fam("nu") / x).sum(0)
                   + (Fj * dFh * fam("nu", h, j) / x).sum(0) + Fj * Fh * fam("tau"))
        values = {
            "M": Fj * Fh * fam("g"), "u": u * Fj * Fh * fam("g"), "Haux_u": s * u * Fj * Fh * fam("g"),
            **{f"R{2 * k}": s**k * Fj * Fh * fam("g") for k in range(1, 5)}, "Y": Fj * Fh * fam("y"), "T": T,
            "Haux_T": s * T - Fj * (Fh * fam("beta") + (dFh * fam("t") / x).sum(0)) / 8,
            "Haux_Y": Fj * Fh * (s * fam("y") - fam("v") / 8),
        }[name]
        return self.Q.integrate(values, float(tj[1] + th[1]) / 2)


@pytest.mark.slow
def test_blocks_against_slice_quadrature(blocks, small_table, small_seeds):
    """Independent route: gauge-slice integrands and singular-value quadrature (no Wick, no lifting)."""
    oracle = SliceIntegrands(small_table, small_seeds.norm)
    T = trials()
    sample = [(0, 0), (2, 2), (1, 3), (4, 6), (9, 30), (5, 26), (15, 22), (36, 41), (12, 40)]
    for name in BLOCKS:
        scale = max(abs(float(blocks[name][j, h])) for j, h in sample) or 1.0
        for j, h in sample:
            exact = float(blocks[name][j, h])
            assert oracle.element(name, T[j], T[h]) == pytest.approx(exact, rel=1e-6, abs=1e-9 * scale), (name, j, h)


@pytest.mark.slow
def test_unequal_longitudinal_widths_against_slice_quadrature(small_table, small_seeds):
    """Trial functions of different longitudinal widths b, d pair with the overlap sqrt(2 sqrt(bd)/(b+d)) of
    normalized one-dimensional Gaussians (ball arithmetic: the overlap is irrational).  The isotropic width
    2/3 is not dyadic: its anisotropy (a - b)/2 must enter the balls as an exact zero."""
    widths = ((Fraction(2), Fraction(1)), (Fraction(3, 2), Fraction(1, 2)), (Fraction(2, 3), Fraction(2, 3)))
    T = trials(widths, ((0, 0, 0), (1, 0, 0)))
    balls = Assembler(T, small_table, small_seeds.norm, Reals(128))
    oracle = SliceIntegrands(small_table, small_seeds.norm, widths)
    sample = [(0, 14), (2, 16), (1, 20), (9, 27), (15, 15), (3, 24), (6, 13), (0, 28), (17, 30), (29, 29), (35, 41)]
    for name in BLOCKS:
        B = balls.matrix(name)
        assert B.overlaps(B.transpose()), name
        scale = max(abs(float(B[j, h].mid())) for j, h in sample) or 1.0
        for j, h in sample:
            b, d = T[j][1], T[h][1]
            overlap = (2 * float(b * d) ** 0.5 / float(b + d)) ** 0.5
            assert oracle.element(name, T[j], T[h]) * overlap == pytest.approx(float(B[j, h].mid()), rel=1e-6,
                                                                                abs=1e-9 * scale), (name, j, h)
