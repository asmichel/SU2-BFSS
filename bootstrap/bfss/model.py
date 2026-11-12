"""SU(2) BFSS in the paper's notation.

T_a = sigma_a / sqrt2, tr(T_a T_b) = delta_ab, f_abc = sqrt2 eps_abc, g = hbar = 1.  The fields are the positions
X_ai, the momenta P_ai and the Majorana fermions theta^a_alpha (color a, vector index i, spinor index alpha); the
matrices are X_i = X_ai T_a.  H = T + u + Y (paper Eqs. (1)-(3)) with T = 1/2 sum P_ai^2, the bosonic potential
u = sum_{i<j} sum_a (X_i x X_j)_a^2 and the Yukawa term Y = sum_ai X_ai K_ai, K_ai = (i / sqrt2) eps_abc
Gamma_i theta^b theta^c.  All operators are exact polynomials (kinematics.Poly); all are cached.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from .kinematics import IDENTITY, SQRT2, Poly, boson, contract, fermion, psum, star

PAIRS = tuple(combinations(range(9), 2))     # spatial pairs i < j, in this order everywhere
IMAG, I_SQRT2, MINUS_I = (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, -1, 0)    # i, i sqrt2, -i


# These Gamma_i are not the paper's (App. A, Eq. (A2)): Gamma_1...Gamma_9 = -1 here and +1 there, so the two sets
# are inequivalent.  They are equivalent to minus the paper's, a sign X -> -X absorbs, so every moment the paper
# reports is unchanged.
def _gamma_matrices() -> np.ndarray:
    f = {"I": np.eye(2, dtype=np.int64), "X": np.array([[0, 1], [1, 0]]),
         "Z": np.array([[1, 0], [0, -1]]), "J": np.array([[0, 1], [-1, 0]])}
    out = []
    for label in ("IIIX", "IIIZ", "IIJJ", "IJXJ", "XJZJ", "ZJZJ", "JIZJ", "JXXJ", "JZXJ"):
        m = f[label[0]]
        for s in label[1:]:
            m = np.kron(m, f[s])
        out.append(m)
    return np.asarray(out, dtype=np.int64)


GAMMA = _gamma_matrices()                    # (9, 16, 16), real symmetric signed permutations


def gamma_form(indices) -> np.ndarray:
    """Gamma_{i1} ... Gamma_{ik} for increasing indices (identity for none)."""
    m = np.eye(16, dtype=np.int64)
    for i in indices:
        m = m @ GAMMA[i]
    return m


def perm_entries(m: np.ndarray):
    """(alpha, beta, sign) for a signed permutation matrix: the one nonzero per row."""
    return tuple((a, int(np.flatnonzero(m[a])[0]), int(m[a, np.flatnonzero(m[a])[0]])) for a in range(m.shape[0]))


def eps(a: int, b: int, c: int) -> int:
    """Levi-Civita symbol eps_abc."""
    if len({a, b, c}) < 3:
        return 0
    return -1 if (int(a > b) + int(a > c) + int(b > c)) % 2 else 1


ADJ_PAIRS = tuple((a, b, c, eps(a, b, c)) for a in range(3) for b, c in permutations(range(3), 2) if eps(a, b, c))


# ---- elementary fields
@lru_cache(maxsize=None)
def X(a: int, i: int) -> Poly:
    """The position X_ai as a polynomial."""
    return Poly.mono(((boson(a, i),), (), 0))


@lru_cache(maxsize=None)
def P(a: int, i: int) -> Poly:
    """The momentum P_ai = -i d/dX_ai as a polynomial."""
    return Poly.mono(((), (boson(a, i),), 0))


@lru_cache(maxsize=None)
def theta(a: int, alpha: int) -> Poly:
    """The Majorana fermion theta^a_alpha as a polynomial."""
    return Poly.mono(((), (), 1 << fermion(a, alpha)))


ONE = Poly.mono(IDENTITY)


@lru_cache(maxsize=None)
def positions() -> tuple[Poly, ...]:
    """All positions X_ai, component 3i + a."""
    return tuple(X(a, I) for I in range(9) for a in range(3))


@lru_cache(maxsize=None)
def momenta() -> tuple[Poly, ...]:
    """All momenta P_ai, component 3i + a."""
    return tuple(P(a, I) for I in range(9) for a in range(3))


# ---- scalars
@lru_cache(maxsize=None)
def R2() -> Poly:
    """R^2 = sum X_ai^2."""
    return contract(positions(), positions())


@lru_cache(maxsize=None)
def R4() -> Poly:
    """R^4 = R^2 R^2."""
    return star(R2(), R2())


@lru_cache(maxsize=None)
def T() -> Poly:
    """Kinetic energy T = 1/2 sum P_ai^2."""
    return contract(momenta(), momenta()).scale(den=2)


@lru_cache(maxsize=None)
def D() -> Poly:
    """Dilatation D = 1/2 sum (X_ai P_ai + P_ai X_ai)."""
    xs, ps = positions(), momenta()
    return (contract(xs, ps, dagger_left=False) + contract(ps, xs, dagger_left=False)).scale(den=2)


@lru_cache(maxsize=None)
def cross() -> tuple[Poly, ...]:
    """(X_i x X_j)_a = eps_abc X_bi X_cj, component 3 pair + a: [X_i, X_j] = i sqrt2 sum_a (X_i x X_j)_a T_a."""
    return tuple(psum(star(X(b, I), X(c, J)).scale(num=s) for aa, b, c, s in ADJ_PAIRS if aa == a)
                 for I, J in PAIRS for a in range(3))


@lru_cache(maxsize=None)
def u() -> Poly:
    """The bosonic potential u = sum_{i<j} sum_a (X_i x X_j)_a^2, paper Eq. (2)."""
    return contract(cross(), cross())


@lru_cache(maxsize=None)
def K() -> tuple[Poly, ...]:
    """K_ai = (i / sqrt2) eps_abc (Gamma_i)_{alpha beta} theta^b_alpha theta^c_beta (paper Eq. (3)), component 3i + a."""
    out = []
    for I in range(9):
        ent = perm_entries(GAMMA[I])
        for a in range(3):
            out.append(psum(star(theta(b, al), theta(c, be)).scale(I_SQRT2, num=s * g, den=2)
                            for aa, b, c, s in ADJ_PAIRS if aa == a for al, be, g in ent))
    return tuple(out)


@lru_cache(maxsize=None)
def Y() -> Poly:
    """The Yukawa term Y = sum_ai X_ai K_ai, paper Eq. (3)."""
    return contract(positions(), K(), dagger_left=False)


@lru_cache(maxsize=None)
def H() -> Poly:
    return T() + u() + Y()


# ---- symmetry generators and supercharges
@lru_cache(maxsize=None)
def gauss() -> tuple[Poly, ...]:
    """Gauss generators gauss_a = i sqrt2 (L_a - G_a), with L_a and G_a of the paper's Eqs. (4) and (6):
    sqrt2 eps_abc X_b.P_c - (i / sqrt2) eps_abc theta^b.theta^c."""
    out = []
    for a in range(3):
        terms = []
        for aa, b, c, s in ADJ_PAIRS:
            if aa != a:
                continue
            terms += [star(X(b, I), P(c, I)).scale(SQRT2, num=s) for I in range(9)]
            terms += [star(theta(b, al), theta(c, al)).scale(I_SQRT2, num=-s, den=2) for al in range(16)]
        out.append(psum(terms))
    return tuple(out)


@lru_cache(maxsize=None)
def S_rot() -> tuple[Poly, ...]:
    """Bosonic rotations S_ij = sum_a (X_ai d_aj - X_aj d_ai) = i sum_a (X_ai P_aj - X_aj P_ai), with d_ai the
    derivative by X_ai (paper, after Eq. (8))."""
    return tuple(psum(star(X(a, I), P(a, J)) - star(X(a, J), P(a, I)) for a in range(3)).scale(IMAG) for I, J in PAIRS)


@lru_cache(maxsize=None)
def R_rot() -> tuple[Poly, ...]:
    """Fermionic rotations R_ij = (1/4) theta^a Gamma_i Gamma_j theta^a (paper Eq. (8))."""
    return tuple(psum(star(theta(a, al), theta(a, be)).scale(num=g, den=4)
                      for a in range(3) for al, be, g in perm_entries(gamma_form(pair)))
                 for pair in PAIRS)


@lru_cache(maxsize=None)
def J_rot() -> tuple[Poly, ...]:
    """Rotation generators J_ij = -i (R_ij + S_ij) (paper, App. B), Hermitian."""
    return tuple((r + s).scale(MINUS_I) for r, s in zip(R_rot(), S_rot()))


def _C(vector, al: int) -> Poly:
    """sum_{i,a} C^alpha_ai vector_{3i+a}, with C^alpha_ai = sum_beta (Gamma_i)_{alpha beta} theta^a_beta
    (paper Eq. (5))."""
    return psum(star(vector[3 * I + a], theta(a, be)).scale(num=g)
                for I in range(9) for aa, be, g in perm_entries(GAMMA[I]) if aa == al for a in range(3))


@lru_cache(maxsize=None)
def CP() -> tuple[Poly, ...]:
    """The first term of the supercharge, sum C^alpha_ai P_ai (paper Eq. (5))."""
    return tuple(_C(momenta(), al) for al in range(16))


@lru_cache(maxsize=None)
def V() -> tuple[Poly, ...]:
    """V_alpha = sqrt2 sum_{i<j} (Gamma_i Gamma_j)_{alpha beta} eps_abc X_bi X_cj theta^a_beta (paper Eq. (5))."""
    out = []
    for al in range(16):
        terms = []
        for I, J in PAIRS:
            be, g = next((b, s) for a_, b, s in perm_entries(gamma_form((I, J))) if a_ == al)
            for a, b, c in permutations(range(3)):
                terms.append(star(star(X(b, I), X(c, J)), theta(a, be)).scale(SQRT2, num=eps(a, b, c) * g))
        out.append(psum(terms))
    return tuple(out)


@lru_cache(maxsize=None)
def Q() -> tuple[Poly, ...]:
    return tuple(a + b for a, b in zip(CP(), V()))

