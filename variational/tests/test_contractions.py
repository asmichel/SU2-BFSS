"""Exact pointwise contractions (bfss.contract) against independent evaluations."""

from dataclasses import dataclass
from fractions import Fraction

import flint
import numpy as np
import pytest
import scipy.sparse as sp

from bfss.algebra import invariants
from bfss.contract import FAMILY_SHAPE, ModularContext, PointEvaluator, mm, primes_1_mod_8
from bfss.fiber import TRIVIAL, OrbitOperator
from bfss.fock import I, FermionOperator
from bfss.model import COLORS, DIMS, MODEL, SPINOR
from conftest import SMALL_CORES
from fock_oracle import FockVector, apply, orbit_tensor

POINT = [Fraction(13, 10), Fraction(7, 10), Fraction(21, 10)]


def family_value(table, f, j, h, x):
    """Exact family polynomial of the core pair (j, h) evaluated at the slice point x."""
    ft = table.families[f]
    N, row = ft.lookup()[(j, h)]
    g = [c * c for c in x]
    e = invariants(g)
    mono = lambda m: e[0] ** m[0] * e[1] ** m[1] * e[2] ** m[2]
    coeffs = ft.coefficients(N, row)
    if ft.diagonal:
        return [sum(c * g[a] ** k * mono(m) for (k, m), c in zip(ft.basis(N), coeffs)) for a in range(3)]
    return sum(c * mono(m) for m, c in zip(ft.basis(N), coeffs))


@pytest.fixture(scope="module")
def oracle(companion):
    """Covariants Y^p S (p <= 3) and all their Cartesian derivatives in the full Fock space."""
    data = companion["fermion-data.json"]
    S = orbit_tensor(data["singlet"]["tensor"], 84)
    x = POINT
    D = [[x[a] if i == a else 0 for i in range(DIMS)] for a in range(COLORS)]
    K = MODEL.K
    Y = sum((K[a][a] * x[a] for a in range(COLORS)), FermionOperator())
    coords = [(a, i) for a in range(COLORS) for i in range(DIMS)]
    Phi, dPhi = [S], [{mu: FockVector([], []) for mu in coords}]
    for _ in range(3):
        dPhi.append({mu: apply(K[mu[0]][mu[1]], Phi[-1]) + apply(Y, dPhi[-1][mu]) for mu in coords})
        Phi.append(apply(Y, Phi[-1]))
    frakb = {nu: sum((MODEL.B((a, a), nu) * x[a] for a in range(COLORS)), FermionOperator()) for nu in coords}
    frakV = sum((MODEL.c(al, D) * MODEL.V(al, D) for al in range(SPINOR)), FermionOperator()) * I
    return dict(Phi=Phi, dPhi=dPhi, frakb=frakb, frakV=frakV, coords=coords)


def direct(o, f, j, h):
    Phi, dPhi, x = o["Phi"], o["dPhi"], POINT
    if f == "g":
        return Phi[j].inner(Phi[h])
    if f == "y":
        return Phi[j].inner(Phi[h + 1])
    if f == "nu":
        return [float(x[a]) * Phi[j].inner(dPhi[h][(a, a)]) for a in range(3)]
    if f == "tau":
        return sum(dPhi[j][mu].inner(dPhi[h][mu]) for mu in o["coords"])
    if f == "beta":
        return sum(Phi[j].inner(apply(o["frakb"][nu], dPhi[h][nu])) for nu in o["coords"])
    if f == "t":
        return [float(x[b]) * Phi[j].inner(apply(o["frakb"][(b, b)], Phi[h])) for b in range(3)]
    if f == "v":
        return Phi[j].inner(apply(o["frakV"], Phi[h]))


@pytest.mark.slow
@pytest.mark.parametrize("family", list(FAMILY_SHAPE))
def test_families_match_full_fock_space(small_table, oracle, family):
    """Orbit/K_4 representation + equivariant off-slice derivatives + CRT reproduce brute force."""
    lookup = small_table.families[family].lookup()
    pairs = [(j, h) for j in range(3) for h in range(3) if (j, h) in lookup]
    assert pairs
    Phi = oracle["Phi"]
    for j, h in pairs:
        exact = family_value(small_table, family, j, h, POINT)
        brute = direct(oracle, family, j, h)
        scale = 1e-11 * abs(Phi[j].inner(Phi[j]) * Phi[h + 1].inner(Phi[h + 1])) ** 0.5  # roundoff floor
        np.testing.assert_allclose(np.real(brute), np.array(exact, dtype=float), rtol=1e-9, atol=scale)
        assert np.max(np.abs(np.imag(brute))) < scale + 1e-9 * np.max(np.abs(brute))


def test_polynomials_predict_fresh_points_modulo_a_new_prime(small_table, small_seeds, library):
    """Equivariance: every contraction is a polynomial in s, u, v (times g_a^k) on the whole slice."""
    p = primes_1_mod_8(80)[-1]
    ctx = ModularContext(library, p)
    values = PointEvaluator(ctx, small_seeds.evaluate(ctx), SMALL_CORES)(fresh := (5, 11, 17))
    for f, ft in small_table.families.items():
        for (j, h) in ft.lookup():
            exact = family_value(small_table, f, j, h, fresh)
            residue = lambda q: q.numerator * pow(q.denominator, -1, p) % p
            got = values[f][:, j, h] if ft.diagonal else values[f][j, h]
            want = [residue(v) for v in exact] if ft.diagonal else residue(exact)
            assert np.array_equal(np.atleast_1d(got) % p, np.atleast_1d(want)), (f, j, h)


def test_exact_identities(small_table, small_seeds):
    fam = small_table.families
    degrees = small_table.degrees
    for j, h in fam["nu"].lookup():
        # Euler: sum_a x_a d_aa core_h = d_h core_h, so the trace of nu is d_h g.
        nu = family_value(small_table, "nu", j, h, POINT)
        assert sum(nu) == degrees[h] * family_value(small_table, "g", j, h, POINT)
        # sum_ab x_a x_b B_{aa,bb} = 0 by antisymmetry of B.
        assert sum(family_value(small_table, "t", j, h, POINT)) == 0
    for f in ("g", "tau", "v"):
        for (j, h) in fam[f].lookup():
            assert family_value(small_table, f, j, h, POINT) == family_value(small_table, f, h, j, POINT)


def test_seed_constants_make_U0_orthogonal_to_S(small_table, small_seeds):
    """<S, Y^2 S> = (80/3) s, so U0 = (126 - C2) Y^2 S - 3360 s S is pointwise orthogonal to S."""
    norm = small_seeds.norm
    s = sum(c * c for c in POINT)
    assert family_value(small_table, "g", 0, 2, POINT) * norm == Fraction(80, 3) * s
    assert family_value(small_table, "g", 0, 4, POINT) == 0


def flavor_casimir() -> FermionOperator:
    """sum_AB E_AB E_BA - N^2/4, E_AB = sum_l f_lA^dagger f_lB: the SU(4) Casimir of the flavor index.

    It is the sum of the squared norms of the traceless flavor generators, so on balanced
    occupations its kernel consists exactly of the Spin(6) ~ SU(4) invariants.
    """
    E = lambda A, B: FermionOperator({(2 * (6 * A + l) + 1, 2 * (6 * B + l)): 1 for l in range(6)})
    N = sum((E(A, A) for A in range(4)), FermionOperator())
    return sum((E(A, B) * E(B, A) for A in range(4) for B in range(4)), FermionOperator()) - N * N / 4


def orthogonalize(N, w, p):
    """B = N T with B^T diag(w) B = diag(d) exactly mod p: Gram-Schmidt with diagonal pivoting."""
    G, T = mm(N.T, w[:, None] * N % p, p), np.eye(N.shape[1], dtype=np.int64)
    for i in range(len(G)):
        j = i + np.flatnonzero(G.diagonal()[i:])[0]
        G[[i, j]], T[:, [i, j]] = G[[j, i]], T[:, [j, i]]
        G[:, [i, j]] = G[:, [j, i]]
        f = G[i, i + 1:] * pow(int(G[i, i]), -1, p) % p
        T[:, i + 1:] = (T[:, i + 1:] - np.outer(T[:, i], f)) % p
        G[i + 1:, i + 1:] = (G[i + 1:, i + 1:] - np.outer(G[i + 1:, i], f)) % p
    return mm(N, T, p), G.diagonal().copy()


@dataclass
class Fiber:
    """Orthogonal basis (columns, in K_4-sector coordinates) of the SU(4) invariants of a K_4 sector."""

    basis: np.ndarray
    weight: np.ndarray  # diagonal of its Gram matrix


class FiberContext(ModularContext):
    """Oracle for the orbit representation: in every K_4 sector, only the Spin(6) ~ SU(4) invariants.

    The invariants are the kernel of the flavor Casimir, 2066-dimensional in the trivial sector (the invariant
    subspace F_*, the paper's Eq. (14)) and 1944-dimensional in each other one.  An orthogonal basis of each, exact
    modulo p, replaces the orbit coordinates (``sectors``; ``k4`` keeps the K_4 sectors); operators become dense
    matrices between them, in sparse containers for ``PointEvaluator``.
    """

    def __init__(self, library, p):
        super().__init__(library, p)
        self.k4 = self.sectors
        self.operators = {**self.operators, "Cf": OrbitOperator(library.space, flavor_casimir())}
        self.sectors = {ch: self._fiber(ch) for ch in self.k4}

    def k4_restricted(self, name, src, dst):
        """Matrix of an operator between two K_4 sectors."""
        return self.reduce(self.full(name)[self.k4[dst].rep] @ self.embed[src])

    def _fiber(self, ch) -> Fiber:
        C, sector = self.k4_restricted("Cf", ch, ch).toarray(), self.k4[ch]
        number = np.bitwise_count(self.space.keys[sector.rep])  # conserved: one block per fermion number
        blocks = []
        for n in np.unique(number):
            rows = np.flatnonzero(number == n)
            X, nullity = flint.nmod_mat(C[np.ix_(rows, rows)].tolist(), self.p).nullspace()
            B, d = orthogonalize(np.array(X.tolist(), dtype=np.int64)[:, :nullity], sector.weight[rows], self.p)
            embedded = np.zeros((len(sector), nullity), dtype=np.int64)
            embedded[rows] = B
            blocks.append((embedded, d))
        return Fiber(np.hstack([b for b, _ in blocks]), np.concatenate([d for _, d in blocks]))

    def coordinates(self, V, ch=TRIVIAL):
        """Coordinates on the SU(4)-invariant basis of K_4-sector vectors (columns) lying in its span."""
        f = self.sectors[ch]
        inverse = np.array([pow(int(d), -1, self.p) for d in f.weight], dtype=np.int64)
        return mm(f.basis.T, self.k4[ch].weight[:, None] * V % self.p, self.p) * inverse[:, None] % self.p

    def restricted(self, name, src=TRIVIAL, dst=TRIVIAL):
        """Matrix of an operator between the SU(4) invariants of two K_4 sectors."""
        key = ("fiber", name, src, dst)
        if key not in self._restricted:
            image = self.reduce(self.k4_restricted(name, src, dst) @ self.sectors[src].basis)
            self._restricted[key] = sp.csr_matrix(self.coordinates(image, dst))
        return self._restricted[key]

    def to_sector(self, v, ch=TRIVIAL):
        s = self.k4[ch]
        assert not ((v[s.partner] - s.e1 * v[s.rep]) % self.p).any(), "vector not in sector"
        return self.coordinates(v[s.rep][:, None], ch)[:, 0]


@pytest.mark.slow
def test_full_reduction_to_the_fibre(library, small_seeds):
    """The SU(4) invariants of the K_4 sectors, the kernel of the flavor Casimir, have the dimensions
    of the stabilizer count, (1,36,468,1056,468,36,1) = 2066 in the trivial sector (F_*, the paper's App. D) and
    (0,30,432,1020,432,30,0) = 1944 in the others; computed in their orthogonal bases (FiberContext) instead of
    orbit space, every contraction is unchanged."""
    p = primes_1_mod_8(1)[0]
    orbit, fiber = ModularContext(library, p), FiberContext(library, p)
    for ch, f in fiber.sectors.items():
        w = fiber.k4[ch].weight
        assert np.array_equal(mm(f.basis.T, w[:, None] * f.basis % p, p), np.diag(f.weight))
        k = np.bitwise_count(fiber.space.keys[fiber.k4[ch].rep]) // 4
        counts = [int((f.basis[k == n] != 0).any(axis=0).sum()) for n in range(7)]
        assert counts == ([1, 36, 468, 1056, 468, 36, 1] if ch == TRIVIAL else [0, 30, 432, 1020, 432, 30, 0])
    x = (2, 3, 5)
    a, b = (PointEvaluator(ctx, small_seeds.evaluate(ctx), SMALL_CORES)(x) for ctx in (orbit, fiber))
    for family in FAMILY_SHAPE:
        assert np.array_equal(a[family], b[family]), family
