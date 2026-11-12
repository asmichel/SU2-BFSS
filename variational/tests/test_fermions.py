"""The fermion representation (the paper, Sec. II and App. A): algebra and supercharge identities."""

from fractions import Fraction

import numpy as np
import pytest

from bfss.fock import I, FermionOperator, Q8
from bfss.model import COLORS, DIMS, GAMMA, MODEL, SPINOR

ZERO = FermionOperator()
ONE = FermionOperator.const(1)
# A rational configuration with no special symmetry.
X = [[Fraction(1), 0, Fraction(2), 0, 0, Fraction(1, 3), 0, 0, 0],
     [0, Fraction(-1), Fraction(1), 0, Fraction(2), 0, 0, 0, 0],
     [Fraction(1, 2), 0, 0, Fraction(1), 0, 0, 0, Fraction(-1), 0]]


def anticommutator(a, b):
    return a * b + b * a


def invariants(X):
    G = np.array(X, dtype=object).dot(np.array(X, dtype=object).T)
    s = G.trace()
    return s, (s * s - G.dot(G).trace()) / 2


def test_canonical_anticommutation_relations():
    f = lambda m, dag: FermionOperator.ladder(m, dag)
    for m in (0, 5, 23):
        for n in (0, 5, 23):
            assert anticommutator(f(m, False), f(n, True)) == FermionOperator.const(int(m == n))
            assert anticommutator(f(m, False), f(n, False)) == ZERO


def test_majoranas_are_hermitian_clifford_generators():
    th = MODEL.theta
    sample = [(a, al) for a in range(COLORS) for al in (0, 1, 6, 15)]
    for a, al in sample:
        assert th[a][al].adjoint() == th[a][al]
        for b, be in sample:
            assert anticommutator(th[a][al], th[b][be]) == FermionOperator.const(Fraction(int((a, al) == (b, be))))


def test_gamma_matrices_are_real_symmetric_clifford():
    for i in range(DIMS):
        assert np.array_equal(GAMMA[i], GAMMA[i].T)
        for j in range(DIMS):
            assert np.array_equal(GAMMA[i] @ GAMMA[j] + GAMMA[j] @ GAMMA[i], 2 * (i == j) * np.eye(SPINOR))


def test_yukawa_operators_match_published_ordered_words(companion):
    for comp in companion["fermion-data.json"]["yukawa_components"]["components"]:
        published = FermionOperator.from_words((tuple(w), Q8.from_json(c)) for w, c in comp["entries"])
        assert published == MODEL.K[comp["a"]][comp["i"]]


def test_generators_are_yukawa_commutators():
    """G_ab = (1/18) sum_i [K_ai, K_bi] and R_ij = (1/16) sum_a [K_ai, K_aj] (the paper's Eqs. (4) and (8))."""
    K = MODEL.K
    for (a, b), G in MODEL.color_generators.items():
        assert sum((K[a][i].commutator(K[b][i]) for i in range(DIMS)), ZERO) / 18 == G
    for i, j in [(0, 1), (2, 5), (4, 8)]:
        assert sum((K[a][i].commutator(K[a][j]) for a in range(COLORS)), ZERO) / 16 == MODEL.spatial_generator(i, j)


def test_yukawa_transforms_as_colour_and_spatial_vector():
    """[G_ab, K_ci] = d_bc K_ai - d_ac K_bi and [R_ij, K_ak] = d_jk K_ai - d_ik K_aj (the paper, Sec. II)."""
    K, G = MODEL.K, MODEL.color_generators
    for (a, b) in G:
        for c in range(COLORS):
            expected = K[a][4] * int(b == c) - K[b][4] * int(a == c)
            assert G[a, b].commutator(K[c][4]) == expected
    for i, j in [(0, 3), (2, 7)]:
        R = MODEL.spatial_generator(i, j)
        for k in (0, 3, 7):
            assert R.commutator(K[1][k]) == K[1][i] * int(j == k) - K[1][j] * int(i == k)


def test_supercharge_square_pieces():
    """{C^alpha_ai, C^alpha_bj} = delta_ab delta_ij and V_alpha^2 = u: pieces of Q_alpha^2 (the paper's Eq. (6))."""
    s, u = invariants(X)
    for alpha in (0, 9):
        for mu in [(0, 0), (2, 5)]:
            for nu in [(0, 0), (1, 5), (2, 5)]:
                C = lambda m: MODEL.C(alpha, *m)
                assert anticommutator(C(mu), C(nu)) == FermionOperator.const(int(mu == nu))
        V = MODEL.V(alpha, X)
        assert V * V == FermionOperator.const(u)


@pytest.mark.slow
def test_identities_behind_the_aux_hamiltonian_integration_by_parts():
    """sum_alpha {C^alpha_ai, V_alpha} = 0 and sum_{alpha,a,i} C^alpha_ai d_ai V_alpha = 16 i Y."""
    V = [MODEL.V(alpha, X) for alpha in range(SPINOR)]
    for mu in [(0, 0), (1, 4), (2, 8)]:
        assert sum((anticommutator(MODEL.C(alpha, *mu), V[alpha]) for alpha in range(SPINOR)), ZERO) == ZERO

    def shifted(alpha, mu, h):
        Xs = [row[:] for row in X]
        Xs[mu[0]][mu[1]] += h
        return MODEL.V(alpha, Xs)

    lhs = sum((MODEL.C(alpha, a, i) * (shifted(alpha, (a, i), 1) - shifted(alpha, (a, i), -1)) / 2
               for alpha in range(SPINOR) for a in range(COLORS) for i in range(DIMS)), ZERO)
    Y = sum((MODEL.K[a][i] * X[a][i] for a in range(COLORS) for i in range(DIMS) if X[a][i]), ZERO)
    assert lhs == Y * (16 * I)


def test_B_bilinears_are_antisymmetric_and_antihermitian():
    for mu, nu in [((0, 0), (1, 1)), ((0, 0), (1, 4)), ((2, 2), (0, 1))]:
        B = MODEL.B(mu, nu)
        assert MODEL.B(nu, mu) == -B
        assert B.adjoint() == -B


def test_slice_frakV_is_the_restriction_to_the_gauge_slice():
    """i sum_alpha c_alpha V_alpha at D(x) equals the slice operators' cubic polynomial."""
    x = [Fraction(2), Fraction(-1), Fraction(3)]
    D = [[x[a] if i == a else 0 for i in range(DIMS)] for a in range(COLORS)]
    direct = sum((MODEL.c(alpha, D) * MODEL.V(alpha, D) for alpha in range(SPINOR)), ZERO) * I
    ops = MODEL.slice_operators()
    monomials = [name for name in ops if name.startswith("cV")]
    from_slice = sum((ops[n] * (x[0] ** int(n[2]) * x[1] ** int(n[3]) * x[2] ** int(n[4])) for n in monomials), ZERO)
    assert direct == from_slice
    assert direct.adjoint() == direct
