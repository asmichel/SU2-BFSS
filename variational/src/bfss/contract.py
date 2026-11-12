"""Exact pointwise fermionic contractions of polynomial singlets.

For cores Y^p Lambda_sigma (Lambda_sigma a seed) the matrices of the paper's Eq. (22) need a few
fermionic contractions, each an equivariant polynomial in X, hence fixed by its
values on the gauge slice (d_ai = d/dX_ai):

    g    = <core_j, core_h>                          invariant
    y    = <core_j, Y core_h>                        invariant
    nu_a = <core_j, x_a d_aa core_h>                 diagonal of an equivariant matrix
    tau  = sum_ai <d_ai core_j, d_ai core_h>         invariant (full 27-dimensional gradient)
    beta = <core_j, frakB core_h>                    invariant, frakB = sum X_ai B_{ai,bk} d_bk
    t_b  = sum_a x_a x_b <core_j, B_{aa,bb} core_h>  diagonal of an equivariant matrix
    v    = <core_j, i frakV core_h>                  invariant, frakV = sum_alpha c_alpha(X) V_alpha(X)

Derivatives off the slice follow from the singlet condition (the paper's Eq. (E.6)): d_ai = -R_ai / x_a
for i > 3, and (d_ab, d_ba) = (x_b G_ab + x_a R_ab, x_a G_ab + x_b R_ab) / (x_b^2 - x_a^2) for a < b <= 3
(one-based as in the paper; the code counts from 0, so i >= 3 and b < 3).  Values are computed exactly
modulo several primes at integer sample points, interpolated in the invariant basis s^k u^l v^m (times
x_a^{2j} for diagonal families) and lifted to rationals by Chinese remaindering.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from math import lcm
from pathlib import Path

import flint
import numpy as np

from .algebra import invariants
from .fiber import PLANE_CHARACTER, TRIVIAL, OrbitOperator, OrbitSpace, SliceArithmetic, reduce_mod
from .fock import Q8
from .model import MODEL

LOG = logging.getLogger("bfss")
PLANES = [(0, 1), (0, 2), (1, 2)]


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def primes_1_mod_8(count, below=1 << 20):
    """Primes p = 1 mod 8 (so that zeta_8 exists in F_p) small enough for exact float64 BLAS."""
    out, p = [], below - (below % 8) + 1
    while len(out) < count:
        p -= 8
        if _is_prime(p):
            out.append(p)
    return out


def zeta8(p):
    g = next(g for g in range(2, p) if pow(g, (p - 1) // 2, p) != 1)
    return pow(g, (p - 1) // 8, p)


# ---------------------------------------------------------------------------
# Invariant monomial bases
# ---------------------------------------------------------------------------
def invariant_monomials(N):
    """Exponents (k, l, m) with k + 2l + 3m = N: the basis s^k u^l v^m of degree N in G."""
    return [(N - 2 * k - 3 * l, k, l) for l in range(N // 3 + 1) for k in range((N - 3 * l) // 2 + 1)]


def diagonal_monomials(N):
    """Basis g_1^j s^k u^l v^m (j < 3) of diagonal entries of degree-N equivariant matrices."""
    return [(j, m) for j in range(3) if N >= j for m in invariant_monomials(N - j)]


def rref_pivots(M: flint.nmod_mat) -> list[int]:
    """Pivot columns of the reduced row echelon form: the greedy (ordered) independent columns."""
    R, rank = M.rref()
    rows = R.tolist()
    return [next(j for j, x in enumerate(rows[i]) if int(x)) for i in range(rank)]


def mm(A, B, p):
    """(A @ B) mod p for dense operands with entries in [0, p): exact in float64."""
    return np.fmod(A.astype(np.float64) @ B.astype(np.float64), p).astype(np.int64)


def _left_inverse(A, p):
    """L with L A = I over F_p, using a maximal independent set of rows of A."""
    rows = rref_pivots(flint.nmod_mat(A.T.tolist(), p))
    assert len(rows) == A.shape[1], "sample points are not unisolvent"
    inv = flint.nmod_mat(A[rows].tolist(), p).inv()
    L = np.zeros((A.shape[1], A.shape[0]), dtype=np.int64)
    L[:, rows] = np.array([[int(x) for x in row] for row in inv.tolist()])
    return L


# ---------------------------------------------------------------------------
# Operator library (exact) and its reductions modulo a prime
# ---------------------------------------------------------------------------
OPERATORS = "operators-v2.pkl"  # the operator library in a cache directory; the name changes with the operator names


class OperatorLibrary:
    """Exact orbit-space matrices of the named slice operators, cached in the file ``cache`` (OPERATORS in a cache
    directory)."""

    def __init__(self, cache: Path | None = None):
        self.space = OrbitSpace()
        if cache is not None and cache.exists():
            self.ops = pickle.loads(cache.read_bytes())
        else:
            self.ops = {k: OrbitOperator(self.space, v) for k, v in MODEL.slice_operators().items()}
            for op in self.ops.values():
                op.space = None
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(pickle.dumps(self.ops))
        for op in self.ops.values():
            op.space = self.space


class ModularContext(SliceArithmetic):
    """Exact arithmetic of the operator library over F_p, on the full orbit space and on K_4 sectors."""

    def __init__(self, library: OperatorLibrary, p: int):
        super().__init__(library.ops, library.space)
        self.library, self.p, self.zeta = library, p, zeta8(p)

    def scalar(self, q: Q8):
        return q.mod(self.p, self.zeta)

    def image(self, op):
        return op.mod(self.p, self.zeta)

    def reduce(self, a):
        return reduce_mod(a, self.p)

    def to_sector(self, v, ch=TRIVIAL):
        """Coordinates of orbit-space vectors lying in a sector (values at the representatives); checks only that
        each partner orbit holds e1 times its representative's value."""
        s = self.sectors[ch]
        assert not ((v[s.partner] - s.e1 * v[s.rep]) % self.p).any(), "vector not in sector"
        return v[s.rep]


# ---------------------------------------------------------------------------
# Pointwise families at one prime
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Core:
    """Core Y^power Lambda_seed of trial wavefunctions."""

    seed: int
    power: int


class PointEvaluator:
    """Exact (mod p) values of all contraction families for a list of cores at sample points."""

    def __init__(self, ctx: ModularContext, seed_values, cores):
        self.ctx, self.p, self.cores = ctx, ctx.p, cores
        self.max_power = max(c.power for c in cores) + 1
        self.seed_poly = [{m: ctx.to_sector(v) for m, v in sv.terms.items()} for sv in seed_values]
        assert all(sv.grade == 0 for sv in seed_values), "seeds must be real"
        self.weight = ctx.sectors[TRIVIAL].weight
        self.cols = [(c.power, c.seed) for c in cores]

    def _mm(self, A, B):
        return mm(A, B, self.p)

    def _gram(self, U, V, ch=TRIVIAL, grade=0):
        W = (self.ctx.sectors[ch].weight[:, None] * U) % self.p
        G = self._mm(W.T, V)
        return (-G) % self.p if grade else G

    def _sp(self, M, V):
        return (M @ V) % self.p

    def __call__(self, x):
        p, ctx = self.p, self.ctx
        x = [int(a) for a in x]
        inv = lambda a: pow(a % p, -1, p)
        mono = lambda m: pow(x[0], m[0], p) * pow(x[1], m[1], p) * pow(x[2], m[2], p) % p
        nseeds = len(self.seed_poly)
        dim = len(self.weight)
        B0 = np.zeros((dim, nseeds), dtype=np.int64)
        dB = np.zeros((3, dim, nseeds), dtype=np.int64)
        for s, poly in enumerate(self.seed_poly):
            for m, v in poly.items():
                B0[:, s] = (B0[:, s] + mono(m) * v) % p
                for a in range(3):
                    if m[a]:
                        dB[a, :, s] = (dB[a, :, s] + m[a] * mono(m[:a] + (m[a] - 1,) + m[a + 1:]) % p * v) % p
        K = [ctx.restricted(f"K{a}{a}") for a in range(3)]
        Y = ctx.reduce(x[0] * K[0] + x[1] * K[1] + x[2] * K[2])
        cores, dcores = [B0], [[dB[a]] for a in range(3)]  # Y^q Lambda_sigma and d_a of it, q = 0, 1, ...
        for _ in range(self.max_power):
            for a in range(3):
                dcores[a].append((self._sp(K[a], cores[-1]) + self._sp(Y, dcores[a][-1])) % p)
            cores.append(self._sp(Y, cores[-1]))
        sel = lambda stack, shift=0: np.stack([stack[q + shift][:, s] for q, s in self.cols], axis=1)
        V, Vp = sel(cores), sel(cores, 1)
        Da = [sel(dcores[a]) for a in range(3)]

        out = {"g": self._gram(V, V), "y": self._gram(V, Vp)}
        out["nu"] = np.stack([x[a] * self._gram(V, Da[a]) % p for a in range(3)])
        tau = sum(self._gram(Da[a], Da[a]) for a in range(3))
        for b in range(3):
            tau -= inv(x[b] ** 2) * self._gram(V, self._mm_dense(f"RR{b}", V)) % p
        beta = -sum(self._gram(self._sp(self._frakb(b, b, x), V), Da[b]) for b in range(3))
        BRx = sum(x[a] * inv(x[b]) % p * ctx.restricted(f"BR{a}{b}") for a in range(3) for b in range(3))
        beta -= self._gram(V, self._dense_apply(BRx, V))
        for (b, k) in PLANES:
            ch = PLANE_CHARACTER[(b, k)]
            Gv = self._sp(ctx.restricted(f"G{b}{k}", TRIVIAL, ch), V)
            Rv = self._sp(ctx.restricted(f"R{b}{k}", TRIVIAL, ch), V)
            gr = ctx.grade(f"G{b}{k}")
            d = inv(x[k] ** 2 - x[b] ** 2)
            dbk = (x[k] * Gv + x[b] * Rv) * d % p
            dkb = (x[b] * Gv + x[k] * Rv) * d % p
            tau += self._gram(dbk, dbk, ch, gr) + self._gram(dkb, dkb, ch, gr)
            beta -= self._gram(self._sp(self._frakb(b, k, x, ch), V), dbk, ch, gr)
            beta -= self._gram(self._sp(self._frakb(k, b, x, ch), V), dkb, ch, gr)
        out["tau"], out["beta"] = tau % p, beta % p
        out["t"] = np.stack([x[b] * self._gram(V, self._sp(self._frakb(b, b, x), V)) % p for b in range(3)])
        cVx = sum(mono(tuple(map(int, name[2:]))) * ctx.restricted(name) for name in ctx.operators if name.startswith("cV"))
        out["v"] = self._gram(V, self._dense_apply(cVx, V))
        return out

    def _frakb(self, b, k, x, ch=TRIVIAL):
        """frakb_{bk}(D(x)) = sum_a x_a B_{(a,a),(b,k)} from the trivial sector to sector ch."""
        names = [(a, f"B{a}{b}{k}") for a in range(3) if f"B{a}{b}{k}" in self.ctx.operators]
        return self.ctx.reduce(sum(x[a] * self.ctx.restricted(n, TRIVIAL, ch) for a, n in names))

    def _dense_apply(self, M, V):
        return self._mm(self.ctx.reduce(M).toarray(), V)

    def _mm_dense(self, name, V):
        return self._dense_apply(self.ctx.restricted(name), V)


# ---------------------------------------------------------------------------
# Interpolation, Chinese remaindering and the exact table
# ---------------------------------------------------------------------------
# family -> (diagonal?, degree offset, parity of d_j + d_h)
FAMILY_SHAPE = {
    "g": (False, 0, 0),
    "y": (False, 1, 1),
    "nu": (True, 0, 0),
    "tau": (False, -2, 0),
    "beta": (False, 0, 0),
    "t": (True, 2, 0),
    "v": (False, 3, 1),
}


def sample_points(count):
    """Integer points 0 < x1 < x2 < x3 of the gauge slice on distinct rays (primitive triples), by size.

    Families are homogeneous, so proportional points carry no new information; three
    values of a diagonal family at one point fix its three invariant components there.
    """
    from math import gcd
    pts, top = [], 3
    while len(pts) < count:
        pts += [(a, b, top) for b in range(2, top) for a in range(1, b) if gcd(gcd(a, b), top) == 1]
        top += 1
    return pts[:count]


def points_needed(groups):
    """Number of sample points: the largest invariant basis among all family degrees, plus 8 spare points.

    The spares replace points whose rows are dependent; points that no interpolation uses are not evaluated.
    """
    return max(len(invariant_monomials(N)) for g in groups.values() for N in g) + 8


def default_workers(entries, dim=3967):
    """Worker processes allowed by CPU count and currently available memory (``entries``: float64 numbers
    a worker holds: the coefficients of one prime and the values at one point)."""
    import os
    per_worker = 0.7e9 + 1.5 * entries * 8 + 3 * dim * dim * 8
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    available = next(int(l.split()[1]) * 1024 for l in meminfo if l.startswith("MemAvailable"))
    return max(1, min((os.cpu_count() or 2) - 1, int(0.5 * available / per_worker)))


def basis_values(points, N, diagonal):
    """Rows: sample points (times the three diagonal entries); columns: degree-N basis monomials."""
    rows = []
    for x in points:
        g = [c * c for c in x]
        e = invariants(g)
        mono = lambda m: e[0] ** m[0] * e[1] ** m[1] * e[2] ** m[2]
        if diagonal:
            rows += [[g[a] ** j * mono(m) for j, m in diagonal_monomials(N)] for a in range(3)]
        else:
            rows.append([mono(m) for m in invariant_monomials(N)])
    return rows


def rational_reconstruction(r, M, den_bits=64):
    """a/b with a = b r (mod M), b < 2^den_bits and |a| < M / 2^(den_bits + 1); None if absent."""
    bound_b = 1 << den_bits
    bound_a = M >> (den_bits + 1)
    r0, r1, s0, s1 = M, r % M, 0, 1
    while r1 > bound_a:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) >= bound_b:
        return None
    return Fraction(r1, s1) if s1 > 0 else Fraction(-r1, -s1)


@dataclass
class FamilyTable:
    """Exact polynomial coefficients of one contraction family, grouped by degree N in G.

    ``pairs[N]`` lists ordered core-index pairs (j, h); row i of ``numerators[N]`` divided
    by ``denominators[N]`` holds the coefficients of pair i on ``invariant_monomials(N)`` (or
    ``diagonal_monomials(N)``).
    """

    name: str
    diagonal: bool
    pairs: dict = field(default_factory=dict)
    numerators: dict = field(default_factory=dict)
    denominators: dict = field(default_factory=dict)

    def coefficients(self, N, row) -> list[Fraction]:
        return [Fraction(int(x), self.denominators[N]) for x in self.numerators[N][row]]

    def basis(self, N):
        return diagonal_monomials(N) if self.diagonal else invariant_monomials(N)

    def lookup(self):
        return {(int(j), int(h)): (N, i) for N, P in self.pairs.items() for i, (j, h) in enumerate(P)}


def family_degrees(name, degrees):
    """Ordered pairs grouped by the G-degree N of the family's polynomial."""
    diagonal, offset, parity = FAMILY_SHAPE[name]
    out = {}
    for j, dj in enumerate(degrees):
        for h, dh in enumerate(degrees):
            if (dj + dh) % 2 == parity and dj + dh + offset >= 0:
                out.setdefault((dj + dh + offset) // 2, []).append((j, h))
    return {N: np.array(P) for N, P in out.items()}


_WORKER = {}


def _init_worker(cache, seeds_json):
    from .covariants import SeedSet
    _WORKER["library"] = OperatorLibrary(cache)
    _WORKER["seeds"] = SeedSet.from_json(seeds_json)


def _prime_residues(p, cores, degrees, points, groups):
    """Interpolated coefficients of every family modulo one prime.

    Interpolation is linear in the values, so the values at each point are folded into the
    coefficients at once (the left inverse's columns of that point): a worker holds the values
    at one point only.  float64 stays exact: entries below p < 2^20, three products per step.
    Points to which every left inverse gives zero weight are not evaluated.
    """
    library, seeds = _WORKER["library"], _WORKER["seeds"]
    ctx = ModularContext(library, p)
    evaluator = PointEvaluator(ctx, seeds.evaluate(ctx), cores)
    inverse = {(f, N): _left_inverse((np.array(basis_values(points, N, FAMILY_SHAPE[f][0]), dtype=object) % p)
                                     .astype(np.int64), p) for f, g in groups.items() for N in g}
    out = {f: {N: np.zeros((inverse[f, N].shape[0], len(P))) for N, P in g.items()} for f, g in groups.items()}
    rows = lambda f, i: slice(3 * i, 3 * i + 3) if FAMILY_SHAPE[f][0] else slice(i, i + 1)
    for i, x in enumerate(points):
        if not any(L[:, rows(f, i)].any() for (f, _), L in inverse.items()):
            continue
        values = evaluator(x)
        for f, g in groups.items():
            diagonal = FAMILY_SHAPE[f][0]
            for N, P in g.items():
                data = values[f][:, P[:, 0], P[:, 1]] if diagonal else values[f][None, P[:, 0], P[:, 1]]
                acc = out[f][N]
                acc += inverse[f, N][:, rows(f, i)].astype(np.float64) @ data.astype(np.float64)
                np.fmod(acc, p, out=acc)
    return {f: {N: a.astype(np.int64) for N, a in g.items()} for f, g in out.items()}


def contract(seeds, cores: list[Core], cache: Path, workers=None, max_primes=64):
    """Exact contraction families for all ordered pairs of cores (see module docstring).

    ``cache`` holds the operator library.  Primes are processed in parallel batches
    until the Chinese-remainder reconstruction stabilizes, verified on the last prime.
    """
    degrees = [seeds.seeds[c.seed].degree + c.power for c in cores]
    groups = {f: family_degrees(f, degrees) for f in FAMILY_SHAPE}
    points = sample_points(points_needed(groups))
    entries = sum(len(diagonal_monomials(N) if FAMILY_SHAPE[f][0] else invariant_monomials(N)) * len(P)
                  for f, g in groups.items() for N, P in g.items()) + 11 * len(cores) ** 2
    workers = workers or default_workers(entries)
    LOG.info("contracting %d cores at %d points on %d workers", len(cores), len(points), workers)
    OperatorLibrary(cache)  # build the cache once before starting workers
    plist, used = primes_1_mod_8(max_primes), []
    residues = {f: {N: [] for N in g} for f, g in groups.items()}
    while len(used) < len(plist):
        batch = plist[len(used): len(used) + max(workers, 2)]
        results = _run_parallel(_prime_residues, [(p, cores, degrees, points, groups) for p in batch], workers,
                                (cache, seeds.to_json()))
        for r in results:
            for f, g in groups.items():
                for N in g:
                    residues[f][N].append(r[f][N])
        used += batch
        table, complete = _reconstruct(groups, residues, used)
        LOG.info("%d primes: reconstruction %s", len(used), "verified" if complete else "incomplete")
        if complete:
            return ContractionTable(cores, degrees, table)
    raise RuntimeError("rational reconstruction did not stabilize within max_primes")


BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run_parallel(fn, args, workers, init_args):
    """Map over argument tuples in spawned single-threaded-BLAS workers (or inline for one worker)."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    if workers == 1:
        _init_worker(*init_args)
        return [fn(*a) for a in args]
    saved = {k: os.environ.get(k) for k in BLAS_THREADS}
    os.environ.update(dict.fromkeys(BLAS_THREADS, "1"))
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=init_args) as pool:
            return list(pool.map(fn, *zip(*args)))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reconstruct(groups, residues, primes):
    """CRT on all but the last prime, verified on the last one."""
    if len(primes) < 2:
        return None, False
    check = primes[-1]
    tables = {}
    for f, g in groups.items():
        t = FamilyTable(f, FAMILY_SHAPE[f][0])
        for N, P in g.items():
            *R, last = residues[f][N]  # (basis, pairs) residues per prime
            crt, M = R[0].astype(object), primes[0]
            for p, r in zip(primes[1:-1], R[1:]):
                crt = crt + M * ((r - crt) * pow(M, -1, p) % p)
                M *= p
            out = np.empty(crt.shape, dtype=object)
            for idx, val in np.ndenumerate(crt):
                q = rational_reconstruction(int(val), M)
                if q is None or (q.numerator * pow(q.denominator, -1, check) - int(last[idx])) % check:
                    return None, False
                out[idx] = q
            den = lcm(1, *(q.denominator for q in out.ravel()))
            t.pairs[N], t.denominators[N] = P, den
            t.numerators[N] = np.vectorize(lambda q: q.numerator * (den // q.denominator), otypes=[object])(out.T)
        tables[f] = t
    return tables, True


@dataclass
class ContractionTable:
    """Exact contraction families for a list of cores; the coefficients exclude the seed set's norm factor
    (1/1144 for the published seeds), which the ``Assembler`` applies."""

    cores: list
    degrees: list
    families: dict

    @cached_property
    def index(self) -> dict:
        """Position of each core (seed index, power) in ``cores``."""
        return {(c.seed, c.power): i for i, c in enumerate(self.cores)}
