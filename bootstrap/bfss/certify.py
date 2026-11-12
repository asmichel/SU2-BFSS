"""Rigorous certificates for SDPB bounds.

A solve minimizes c_0 + c.z subject to M_b(z) = F_b^0 + sum_n z_n F_b^n >= 0.  For matrices X_b >= 0 and
r_n = c_n - sum_b <X_b, F_b^n>, every feasible z satisfies
    c_0 + c.z = c_0 - sum_b <X_b, F_b^0> + sum_b <X_b, M_b(z)> + r.z  >=  beta - |r|_1 max_n |z_n|,
with beta = c_0 - sum_b <X_b, F_b^0>.  The certificate matrices (SDPB's primal solution, x_b.txt) are rounded to rationals and
shifted by 1e-60 times the identity (or twice a negative eigenvalue left by SDPB's arithmetic; they
vanish outside an active subspace at the optimum), proven
positive definite by an interval Cholesky factorization, and beta and r are computed exactly.  The bound
holds for every state whose coordinates z (moments) are below Z in size: beta - |r|_1 Z.
Numbers are in Q(sqrt2), as pairs (a, b) = a + b sqrt2.
"""
from __future__ import annotations

import math
from fractions import Fraction as Fr
from pathlib import Path

import mpmath

from .sdp import SDP, ZERO, qadd, qmul

DIGITS = 120
SHIFT = Fr(1, 10 ** 60)


def _read_x(path: Path) -> list:
    words = path.read_text().split()
    n = int(words[0])
    return [Fr(w).limit_denominator(10 ** DIGITS) for w in words[2:2 + n]]


def _certificate(sdp: SDP, out: Path) -> list:
    """X_b from SDPB's x_b: the upper triangle column by column, off-diagonal entries doubled (checked by
    the residual of the equalities, which must be tiny)."""
    Xs = []
    for b, (n, _) in enumerate(sdp.blocks):
        x = _read_x(out / f"x_{b}.txt")
        if len(x) != n * (n + 1) // 2:
            raise ValueError(f"block {b}: {len(x)} entries for size {n}")
        X, k = {}, 0
        for j in range(n):
            for i in range(j + 1):
                X[(i, j)] = (x[k] if i == j else x[k] / 2, Fr(0))
                k += 1
        Xs.append(X)
    return Xs


def _inner(X: dict, M: dict, n: int) -> tuple:
    """<X, A> for symmetric X (upper triangle, rational) and the coordinate-n part of the affine entries M."""
    acc = ZERO
    for (i, j), (c, lin) in M.items():
        v = c if n == 0 else lin.get(n - 1)
        if v is None or v == ZERO:
            continue
        w = X.get((i, j), ZERO)
        acc = qadd(acc, qmul((w[0] * (1 if i == j else 2), w[1] * (1 if i == j else 2)), v))
    return acc


def certify(sdp: SDP, out: Path, Z: float = 1e40) -> dict:
    """Rigorous bound from the SDPB solution in out: beta (exact), |r|_1 and beta - |r|_1 Z (floats)."""
    m = len(sdp.coordinates)
    Xs = _certificate(sdp, out)
    mpmath.mp.dps = 80
    for X, (n, _) in zip(Xs, sdp.blocks):              # shift: SHIFT, or twice a negative rounding eigenvalue
        D = mpmath.matrix(n, n)
        for (i, j), (a, b) in X.items():
            D[i, j] = D[j, i] = mpmath.mpf(a.numerator) / a.denominator + mpmath.mpf(b.numerator) / b.denominator * mpmath.sqrt(2)
        low = min(mpmath.eigsy(D, eigvals_only=True)) if n > 1 else D[0, 0]
        s = max(SHIFT, Fr(mpmath.nstr(-2 * low, 5)) if low < 0 else SHIFT)
        for i in range(n):
            X[(i, i)] = qadd(X.get((i, i), ZERO), (s, Fr(0)))
    margins = [_cholesky_margin(X, n) for X, (n, _) in zip(Xs, sdp.blocks)]
    if min(margins) <= 0:
        raise RuntimeError(f"a certificate matrix is not provably positive definite (block {margins.index(min(margins))})")
    c = [sdp.objective[1].get(k, ZERO) for k in range(m)]
    resid = [qadd(c[k], tuple(-v for v in sum_q(_inner(X, M, k + 1) for X, (_, M) in zip(Xs, sdp.blocks))))
             for k in range(m)]
    beta = qadd(sdp.objective[0], tuple(-v for v in sum_q(_inner(X, M, 0) for X, (_, M) in zip(Xs, sdp.blocks))))
    iv = mpmath.iv                                     # beta - |r|_1 Z in interval arithmetic: rigorous
    iv.prec = 4 * DIGITS
    r2 = iv.sqrt(iv.mpf(2))
    val = lambda q: iv.mpf(q[0].numerator) / q[0].denominator + iv.mpf(q[1].numerator) / q[1].denominator * r2
    r1 = sum((abs(val(r)) for r in resid), iv.mpf(0))
    if r1.b > 1e-30:                                   # not a solution of SDPB's (e.g. an aborted iterate)
        raise RuntimeError(f"certificate matrices violate the equalities by {mpmath.nstr(r1.b, 5)}: no certificate")
    low = (val(beta) - r1 * iv.mpf(Z)).a
    bound = float(low)
    if bound > low:                                    # a lower bound is rounded down, never to nearest
        bound = math.nextafter(bound, -math.inf)
    mpmath.mp.dps = DIGITS
    return {"beta": beta, "beta_digits": mpmath.nstr(val(beta).a, 30), "residual_l1": float(r1.b),
            "bound": bound, "Z": Z, "min_pivot": float(min(margins))}


def sum_q(it) -> tuple:
    acc = ZERO
    for v in it:
        acc = qadd(acc, v)
    return acc


def _cholesky_margin(X: dict, n: int, prec: int = 512):
    """Smallest Cholesky pivot of X as an interval lower bound (> 0 proves positive definiteness)."""
    mpmath.iv.prec = prec
    iv = mpmath.iv
    A = [[iv.mpf(0)] * n for _ in range(n)]
    r2 = iv.sqrt(iv.mpf(2))
    for (i, j), (a, b) in X.items():
        A[i][j] = A[j][i] = iv.mpf(a.numerator) / a.denominator + iv.mpf(b.numerator) / b.denominator * r2
    low = mpmath.inf
    for k in range(n):
        s = A[k][k] - sum((A[k][t] ** 2 for t in range(k)), iv.mpf(0))
        if s.a <= 0:
            return float(s.a)
        low = min(low, s.a)
        d = iv.sqrt(s)
        A[k][k] = d
        for i in range(k + 1, n):
            A[i][k] = (A[i][k] - sum((A[i][t] * A[k][t] for t in range(k)), iv.mpf(0))) / d
    return float(low)
