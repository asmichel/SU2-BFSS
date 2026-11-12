"""Orthonormal coordinates, dilation and the variational problem (the paper's Eqs. (21) and (22)).

Reduced matrices of a basis at its canonical widths (largest b = 1) come from ``blocks``, exactly or as
balls of a working precision.  The ordered Gram-Schmidt process omits
functions with zero residual: for exact matrices we find them from the rank profile of
the Gram matrix modulo a large prime, for balls by a relative tolerance; the remaining
functions are orthonormalized with a blocked Cholesky factorization in multiprecision
arithmetic.  Minimizing the smallest eigenvalue of
U_lambda^-1 H_aux U_lambda = Haux_T + lambda^-3 Haux_u + lambda^-3/2 Haux_Y in the orthonormal frame over the
dilation lambda maximizes the bound [1 - kappa <H_aux>]_+ on F.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import add

import flint
import numpy as np
import scipy.linalg
import scipy.optimize
from flint import arb, arb_mat, fmpz_mat, nmod_mat

from .blocks import OBSERVABLES, Assembler
from .contract import rref_pivots
from .scalars import RATIONALS, Scalars, ball

LOG = logging.getLogger("bfss")
KAPPA = (36 + 8 * 14**0.5) / 25
MERSENNE_61 = (1 << 61) - 1


class ReducedBlocks:
    """Reduced block matrices of a basis at its canonical widths (largest b = 1), in a number system."""

    def __init__(self, basis, table, scalars: Scalars = RATIONALS):
        self.basis, self.scalars = basis, scalars
        self.assembler = Assembler(basis.assembler_trials(table), table, basis.seeds.norm, scalars)

    def matrix(self, name):
        """The block, assembled on every request."""
        M = self.assembler.matrix(name)
        LOG.info("assembled %s (%d functions)", name, len(self.basis))
        return M

    def arb(self, name, subset=None) -> arb_mat:
        """The block as a ball matrix at the current precision (restricted to ``subset``)."""
        M = self.matrix(name)
        M = M if isinstance(M, arb_mat) else arb_mat(M)  # ball blocks are used as they are, not copied
        return M if subset is None else _restricted(M, subset)

    def quadratic_form(self, name, x) -> arb:
        """x^T B x for a ball coefficient vector over the whole basis."""
        X = arb_mat([[v] for v in x])
        return (X.transpose() * (self.arb(name) * X))[0, 0]

    def independent_subset(self) -> list[int]:
        """Greedy ordered set of linearly independent functions: the rank profile of M modulo the prime 2^61 - 1
        (the rational rank profile unless that prime divides a relevant minor).

        With ball arithmetic dependence is numerical and left to the orthonormalization.
        """
        if not self.scalars.exact:
            return list(range(len(self.basis)))
        numerators, _ = self.matrix("M").numer_denom()
        return rref_pivots(nmod_mat(numerators, MERSENNE_61))


def _restricted(M: arb_mat, subset) -> arb_mat:
    """The rows and columns of a ball matrix in ``subset``."""
    rows = M.tolist()
    return arb_mat([[rows[i][j] for j in subset] for i in subset])


def _sub(rows, r0, r1, c0, c1):
    return [row[c0:c1] for row in rows[r0:r1]]


def gram_schmidt(A: arb_mat, nb: int = 96, tolerance: float = 0.0, block: int = 2048):
    """Ordered Gram-Schmidt of a unit-diagonal Gram matrix: the inverse of its Cholesky factor.

    Recursive 2x2 blocking reduces everything but blocks of at most ``nb`` functions to dense
    ball-matrix products.  Arithmetic is multiprecision floating point (ball midpoints):
    interval bounds through the triangular solves of ill-conditioned blocks grow far beyond
    the actual rounding error.  A function whose squared residual falls to ``tolerance`` or
    below is omitted.  The products go through
    ``frame_product``.  Returns (L^{-1} restricted to the kept functions, kept indices).
    """
    n, rows = A.nrows(), A.tolist()
    if n <= nb:
        return _gram_schmidt_small([[x.mid() for x in row] for row in rows], tolerance)
    h = n // 2
    I1, k1 = gram_schmidt(arb_mat(_sub(rows, 0, h, 0, h)), nb, tolerance, block)
    A22 = arb_mat(_sub(rows, h, n, h, n))
    if not k1:
        I2, k2 = gram_schmidt(A22, nb, tolerance, block)
        return I2, [h + j for j in k2]
    B = frame_product(arb_mat([[row[j] for j in k1] for row in rows[h:]]), I1.transpose(), block)  # L21 on k1
    del rows  # the lists of the whole block, released before the second recursion
    I2, k2 = gram_schmidt((A22 - frame_product(B, B.transpose(), block)).mid(), nb, tolerance, block)
    if not k2:
        return I1, k1
    Bk = B.tolist()
    C = (-frame_product(frame_product(I2, arb_mat([Bk[i] for i in k2]), block), I1, block)).tolist()
    top = [r + [arb(0)] * len(k2) for r in I1.tolist()]
    return arb_mat(top + [rc + r2 for rc, r2 in zip(C, I2.tolist())]), k1 + [h + j for j in k2]


def frame_product(X: arb_mat, Y: arb_mat, block: int = 2048) -> arb_mat:
    """Midpoint of the product X Y of two blocks of the Gram-Schmidt recursion.

    The blocks (of the inverse Cholesky factors, and L21 = A21 L11^{-T}) have entries over many binades;
    FLINT's ball product splits such operands into blocks of similar exponents and slows down badly beyond
    a few thousand rows.  Large products therefore go through fixed point at the working precision plus 64 bits and blocked integer
    products, as the frame transform does: the error is below 2^-(prec+50) of max|X| max|Y|, no larger
    than that of the ball product.
    """
    if min(X.nrows(), X.ncols(), Y.ncols()) < 3 * block:
        return (X * Y).mid()
    bits = flint.ctx.prec + 64
    Xz, sx = fixed_point(X, [0] * X.nrows(), [0] * X.ncols(), bits)
    Yz, sy = fixed_point(Y, [0] * Y.nrows(), [0] * Y.ncols(), bits)
    R, K, C = (_spans(m, block) for m in (X.nrows(), X.ncols(), Y.ncols()))
    Xb, Yb = _blocks(Xz, R, K), _blocks(Yz, K, C)
    del Xz, Yz
    rows = []
    for i in range(len(R)):
        parts = [reduce(add, (Xb[i][k] * Yb[k][j] for k in range(len(K)))).tolist() for j in range(len(C))]
        rows += [[x for p in parts for x in p[r]] for r in range(len(parts[0]))]
    return (arb_mat(rows) * arb(2) ** -(sx + sy)).mid()


def _gram_schmidt_small(work, tolerance):
    n = len(work)
    L = [[arb(0)] * n for _ in range(n)]
    kept = []
    for j in range(n):
        s = (work[j][j] - sum((L[j][t] ** 2 for t in kept), arb(0))).mid()
        if not s > tolerance:
            continue
        L[j][j] = s.sqrt().mid()
        for i in range(j + 1, n):
            L[i][j] = ((work[i][j] - sum((L[i][t] * L[j][t] for t in kept), arb(0))) / L[j][j]).mid()
        kept.append(j)
    L = [[L[i][j] for j in kept] for i in kept]
    inv = [[arb(0)] * len(kept) for _ in kept]
    for j in range(len(kept)):
        inv[j][j] = (1 / L[j][j]).mid()
        for i in range(j + 1, len(kept)):
            inv[i][j] = (-sum((L[i][t] * inv[t][j] for t in range(j, i)), arb(0)) / L[i][i]).mid()
    return arb_mat(inv) if kept else arb_mat(0, 0), kept


def to_float(M: arb_mat) -> np.ndarray:
    return np.array([[float(x.mid()) for x in row] for row in M.tolist()])


def fixed_point(M: arb_mat, rows, cols, bits: int):
    """(Z, s) with M_ij 2^(rows_i + cols_j) = Z_ij 2^-s up to rounding, the largest |Z_ij| about 2^bits.

    Exact integer matrices multiply much faster than balls (multimodular, multithreaded);
    the power-of-two row and column scalings keep the entries within a common range.
    Z is returned as rows of integers.
    """
    entries = M.tolist()
    me = lambda i: [(int(m), int(e) + rows[i] + c) for x, c in zip(entries[i], cols) for m, e in [x.mid().man_exp()]]
    s = bits - max((e + m.bit_length() for i in range(len(entries)) for m, e in me(i) if m), default=0)
    out = []
    for i in range(len(entries)):  # row by row, releasing the balls: the peak is one copy of the matrix
        out.append([m << (e + s) if e + s >= 0 else m >> -(e + s) for m, e in me(i)])
        entries[i] = None
    return out, s


def _spans(n, block):
    """Blocks (start, end) of at most ``block`` indices."""
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def _blocks(rows, spans, cols=None, lower=False):
    """Integer matrices of the blocks of a matrix of integer rows: rows in ``spans`` (start, end), columns in
    ``cols`` (default the same); with ``lower`` only the blocks on and below the diagonal."""
    cols = spans if cols is None else cols
    return [[fmpz_mat(_sub(rows, r0, r1, c0, c1)) for c0, c1 in cols[:i + 1 if lower else None]]
            for i, (r0, r1) in enumerate(spans)]


@dataclass
class OrthonormalFrame:
    """Gram-Schmidt frame: functions ``subset`` with phi_A = sum_j tilde psi_j C_jA, C = L^{-T}.

    ``scale`` holds powers of two near the norms of the functions: the frame and the blocks,
    equilibrated by them, have entries of moderate range and are transformed in fixed point.
    """

    subset: list[int]
    Linv: arb_mat
    scale: list[int]
    block: int = 2048

    @cached_property
    def spans(self):
        """Blocks (start, end) of the frame for the integer products."""
        return _spans(len(self.subset), self.block)

    @cached_property
    def fixed(self):
        """L^{-1} in fixed point (its blocks on and below the diagonal), the exponent and the bits."""
        bits = flint.ctx.prec + 64
        Z, s = fixed_point(self.Linv, [0] * len(self.scale), self.scale, bits)
        return _blocks(Z, self.spans, lower=True), s, bits

    def transform(self, B: arb_mat) -> np.ndarray:
        """Matrix of a quadratic form in the orthonormal frame (rounded to float64): L^{-1} B L^{-T}.

        The integer products go in blocks of ``block`` rows, skip the zero blocks of the triangular L^{-1} and
        compute only the lower blocks of the symmetric result, which need only the lower blocks of P = L^{-1} B.
        """
        L, s, bits = self.fixed
        down = [-k for k in self.scale]
        Z, t = fixed_point(B, down, down, bits)
        Bz = _blocks(Z, self.spans)
        del Z
        out, unit = np.empty((len(self.subset),) * 2), arb(2) ** -(2 * s + t)
        for i, (r0, r1) in enumerate(self.spans):
            PT = [reduce(add, (L[i][k] * Bz[k][j] for k in range(i + 1))).transpose() for j in range(i + 1)]
            for j, (c0, c1) in enumerate(self.spans[:i + 1]):  # block (i, j) = (sum_k L_jk P_ik^T)^T
                out[r0:r1, c0:c1] = to_float(arb_mat(reduce(add, (L[j][k] * PT[k] for k in range(j + 1)))) * unit).T
                if j < i:
                    out[c0:c1, r0:r1] = out[r0:r1, c0:c1].T
        return out

    def raw_coefficients(self, z) -> list:
        """Coefficients x_j = (C z)_j of the kept functions only, in ``subset`` order."""
        zc = arb_mat([[arb(float(v))] for v in z])
        y = (self.Linv.transpose() * zc).tolist()
        return [y[k][0] for k in range(len(self.subset))]


def orthonormal_frame(blocks: ReducedBlocks, nb: int = 96, tolerance: float | None = None, omit=()) -> OrthonormalFrame:
    """Ordered Gram-Schmidt frame, computed on the equilibrated Gram matrix.

    With M = D N D, D = diag(M_jj)^{1/2}, the residuals are relative to each function's own norm
    although the norms span many orders of magnitude.  Exactly dependent functions (zero rank
    profile modulo a large prime) are always omitted; ``tolerance`` also omits functions whose
    relative squared residual is not larger than it (default: 0 for exact blocks, 2^(-prec/2) for
    balls of prec bits), and ``omit`` lists functions left out by choice.
    """
    if tolerance is None:
        tolerance = 0.0 if blocks.scalars.exact else 2.0 ** (-blocks.scalars.prec / 2)
    omit = set(omit)
    subset = [j for j in blocks.independent_subset() if j not in omit]
    rows = blocks.arb("M", subset).tolist()
    assert all(x.is_finite() for row in rows for x in row), "non-finite Gram matrix entries"
    d = [rows[i][i].sqrt() for i in range(len(subset))]
    N = arb_mat([[x / (d[i] * d[j]) for j, x in enumerate(row)] for i, row in enumerate(rows)])
    del rows
    Linv, kept = gram_schmidt(N, nb, tolerance)
    scale = [int(e) + int(m).bit_length() for m, e in (d[k].mid().man_exp() for k in kept)]  # 2^scale ~ d
    Linv = arb_mat([[x / d[kept[j]] for j, x in enumerate(row)] for row in Linv.tolist()])
    return OrthonormalFrame([subset[k] for k in kept], Linv, scale)


def lowest_eigenpair(A: np.ndarray, guess=None, block: int = 24, tol: float = 1e-11):
    """Smallest eigenvalue and unit eigenvector of a symmetric positive definite matrix.

    Subspace iteration with A^{-1} (one Cholesky factorization) and Rayleigh-Ritz.  The iteration
    converges with the ratio of the lowest eigenvalue to the (block + 1)-th; profiles reaching far
    along the valley crowd the bottom of the spectrum, and a wide block keeps the iterations few.
    The lowest Ritz value decreases until it reaches the rounding floor of A, about eps |A|, which
    lies far above tol |w| when the eigenvalue is ~1e-6 of the largest: the iteration also stops
    when the Ritz value no longer decreases.
    """
    factor = scipy.linalg.cho_factor(A, lower=True)
    V = np.random.default_rng(0).standard_normal((len(A), block))
    if guess is not None:
        V[:, 0] = guess
    previous = np.inf
    for _ in range(100):
        V = np.linalg.qr(scipy.linalg.cho_solve(factor, V))[0]
        w, U = np.linalg.eigh(V.T @ A @ V)
        V = V @ U
        if abs(previous - w[0]) <= tol * abs(w[0]) or w[0] >= previous:
            break
        previous = w[0]
    return w[0], V[:, 0]


@dataclass
class VariationalResult:
    dilation: float
    eigenvalue: float
    fidelity_bound: float
    x: list  # coefficients of the undilated reduced trial wavefunctions tilde psi_j (full basis order)


def optimize(blocks: ReducedBlocks, bracket=(1 / 16, 16), nb: int = 96, tolerance: float | None = None,
             omit=(), dilation=None) -> VariationalResult:
    """Minimize the smallest eigenvalue of U_lambda^-1 H_aux U_lambda over the dilation (the paper's Eq. (22)).

    A geometric grid over ``bracket`` locates the minimum, which Brent's method then refines.  At a given
    rational ``dilation`` only U_lambda^-1 H_aux U_lambda at that lambda is transformed, one block instead of
    three.
    """
    if dilation is not None and dilation <= 0:
        raise ValueError(f"a dilation is positive, not {dilation}")
    frame = orthonormal_frame(blocks, nb, tolerance, omit)
    LOG.info("orthonormal frame: %d of %d functions", len(frame.subset), len(blocks.basis))
    haux = OBSERVABLES["Haux"]
    if dilation is None:
        parts = {k: frame.transform(blocks.arb(k, frame.subset)) for k in haux.parts}
        LOG.info("transformed %s", ", ".join(parts))
        vector = [None]

        def smallest(lam):
            w, vector[0] = lowest_eigenpair(haux.dilated(parts, lam), vector[0])
            return w

        grid = np.geomspace(*bracket, 17)
        i = int(np.argmin([smallest(lam) for lam in grid]))
        if i in (0, len(grid) - 1):
            LOG.warning("the dilation is optimal at the end of %s", bracket)
        bounds = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        lam = scipy.optimize.minimize_scalar(smallest, bounds=bounds, method="bounded", options={"xatol": 1e-10}).x
        w, v = smallest(lam), vector[0]
    else:
        lam = float(dilation)
        A = sum(ball(dilation) ** e * blocks.arb(k) for k, e in haux.parts.items())
        A = _restricted(A, frame.subset)  # rebinding releases the unrestricted sum
        w, v = lowest_eigenpair(frame.transform(A))
    LOG.info("dilation %.12g: smallest eigenvalue %.12g", lam, w)
    xs = frame.raw_coefficients(v)
    x = [arb(0)] * len(blocks.basis)
    sign = -1 if xs[0] < 0 else 1  # phase convention: the first trial wavefunction has a positive coefficient
    for k, j in enumerate(frame.subset):
        x[j] = sign * xs[k]
    return VariationalResult(lam, w, max(0.0, 1 - KAPPA * w), x)
