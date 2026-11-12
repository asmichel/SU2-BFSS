"""The complete operator basis at level L.

Operators are Weyl symbols: polynomials in X_ai, P_ai (commuting) and theta^a_alpha (Grassmann), of
level 2 #x + 4 #p + 3 #theta (twice the scaling dimension).  SU(2) x Spin(9) acts on symbols by derivations
(the Moyal commutator with a quadratic generator is linear in the variables), read off from the generators of
model.py.  In weight coordinates (complex combinations of the variables diagonalizing a Cartan subalgebra) the
multiplets of an irrep in a grade (#x, #p, #theta) are given by their highest-weight vectors: the kernel of the
simple raising derivations on the monomials of the highest weight.  Kernels are exact (elimination modulo
primes, rational reconstruction) and their dimensions are checked against the character of the algebra.

A multiplet is a row of the level-L hierarchy if its level is at most L and its norm is finite in the
true ground state: its growth exponent e along the flat valley is below 9/2.  e is the leading power of r
when, in a generic frame, the valley coordinate is ~ r, the other coordinates ~ r^-1/2, the massive
momenta ~ r^1/2, the valley and gauge momenta ~ r^-1 (Gauss's law) and the fermions ~ 1.  Cancellations
within a sector are found exactly: the rows are the combinations without terms of growth >= 9/2.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Fr
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from math import lcm, prod
from pathlib import Path

import numpy as np
from scipy import sparse

from . import model as M
from .kinematics import I as QI, SQRT2, Poly, star

NX, NF = 27, 48
TAIL_ROW = 9                      # rows: valley degree (2e) below 9
Z0, Z1 = (Fr(0), Fr(0)), (Fr(1), Fr(0))
CARTAN_PAIRS = ((0, 2), (3, 5), (4, 7), (6, 8))       # spinor weight vectors then mix two real thetas
SIMPLE = ((2, 0, 0, 0, 0), (0, 2, -2, 0, 0), (0, 0, 2, -2, 0), (0, 0, 0, 2, -2), (0, 0, 0, 0, 2))   # doubled
ROOTS = [(2, 0, 0, 0, 0)] + [(0,) + tuple(2 * (k == i) for k in range(4)) for i in range(4)] + \
        [(0,) + tuple(2 * ((k == i) + s * (k == j)) for k in range(4)) for i, j in combinations(range(4), 2)
         for s in (1, -1)]


# ---- Q(i): pairs of Fractions
def zadd(u, v):
    return (u[0] + v[0], u[1] + v[1])


def zmul(u, v):
    return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])


def zconj(u):
    return (u[0], -u[1])


def zinv(u):
    n = u[0] * u[0] + u[1] * u[1]
    return (u[0] / n, -u[1] / n)


def zsum(it):
    re, im = Fr(0), Fr(0)
    for a, b in it:
        re += a
        im += b
    return (re, im)


def zrot(v, k: int):
    """v times i^k."""
    for _ in range(k % 4):
        v = (-v[1], v[0])
    return v


def _rref_exact(rows: list, n: int):
    """Reduced row echelon form over Q(i) (small matrices): (rows, pivot columns)."""
    rows = [list(r) for r in rows]
    piv, r = [], 0
    for c in range(n):
        k = next((i for i in range(r, len(rows)) if rows[i][c] != Z0), None)
        if k is None:
            continue
        rows[r], rows[k] = rows[k], rows[r]
        inv = zinv(rows[r][c])
        rows[r] = [zmul(inv, v) for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != Z0:
                f = rows[i][c]
                rows[i] = [zadd(a, zmul((-f[0], -f[1]), b)) for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    return rows[:r], piv


def _kernel_exact(rows: list, n: int) -> list:
    red, piv = _rref_exact(rows, n)
    out = []
    for f in (c for c in range(n) if c not in piv):
        v = [Z0] * n
        v[f] = Z1
        for r, c in enumerate(piv):
            v[c] = (-red[r][f][0], -red[r][f][1])
        out.append(v)
    return out


def _matmul_exact(A, B):
    return [[zsum(zmul(a, B[k][j]) for k, a in enumerate(row) if a != Z0) for j in range(len(B[0]))] for row in A]


def _inverse_exact(A):
    n = len(A)
    red, piv = _rref_exact([row + [Z1 if i == j else Z0 for j in range(n)] for i, row in enumerate(A)], 2 * n)
    if piv[:n] != list(range(n)):
        raise ValueError("singular matrix")
    return [row[n:] for row in red]


# ---- the group action
VARS = {"x": [M.X(A // 9, A % 9) for A in range(NX)], "p": [M.P(A // 9, A % 9) for A in range(NX)],
        "f": [M.theta(j // 16, j % 16) for j in range(NF)]}


def _derivation(T: Poly, kind: str) -> list:
    """R[k][l]: coefficient of variable l in i [T, variable k]."""
    vs = VARS[kind]
    index = {next(iter(v.terms)): k for k, v in enumerate(vs)}
    R = [[Z0] * len(vs) for _ in vs]
    for k, v in enumerate(vs):
        c = (star(T, v) - star(v, T)).scale(QI)
        for m, (a, b, cc, d) in c.terms.items():
            if b or d:
                raise ValueError("derivation outside Q(i)")
            R[k][index[m]] = (Fr(a, c.den), Fr(cc, c.den))
    return R


def _candidates(kind: str) -> list:
    vec = [tuple(2 * s * (j == i) for j in range(4)) for i in range(4) for s in (1, -1)] + [(0,) * 4]
    spin = list(product((1, -1), repeat=4))
    return [(c,) + v for c in (2, 0, -2) for v in (spin if kind == "f" else vec)]


@lru_cache(maxsize=None)
def action() -> dict:
    """The group action in weight coordinates (cached on disk, keyed by the sources it is computed from)."""
    src = Path(__file__).resolve().parent
    key = hashlib.sha256(b"".join((src / f).read_bytes() for f in ("kinematics.py", "model.py", "basis.py"))).hexdigest()[:16]
    path = src / "__pycache__" / f"action-{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = _action()
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")         # concurrent jobs never read half a file
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(path)
    return out


def _action() -> dict:
    """Weight coordinates and simple root vectors.

    Per kind (x, p, f): 'V' (weight variable k = sum_A V[k][A] variable_A), 'Vinv', 'wt' (doubled weights),
    'bar' (index of the opposite weight), 'conj' (w_k^* = conj[k] w_bar[k]), 'pair' (sum_A V[k][A]
    V[bar k][A], the commutator / anticommutator form) and 'norm' (sum_A |V[k][A]|^2).  'raise' and
    'lower': per simple root, per kind, {k: (l, coefficient)}."""
    gens = [g.scale(SQRT2, den=2) for g in M.gauss()] + list(M.J_rot())    # sqrt2-free
    cartan = [2] + [3 + M.PAIRS.index(pr) for pr in CARTAN_PAIRS]
    out, D = {}, {}
    for kind in ("x", "p", "f"):
        R = [_derivation(g, kind) for g in gens]
        n = len(VARS[kind])
        V, wt = [], []
        for mu in _candidates(kind):                  # left eigenvectors c R_H = (i mu / 2) c
            eqs = [[zadd(R[h][k][l], (Fr(0), -Fr(m, 2))) if k == l else R[h][k][l] for k in range(n)]
                   for h, m in zip(cartan, mu) for l in range(n)]
            ker = _kernel_exact(eqs, n)
            if len(ker) != 1:
                raise RuntimeError(f"weight {mu} of {kind}: eigenspace of dimension {len(ker)}")
            lead = zinv(next(z for z in ker[0] if z != Z0))
            V.append([zmul(lead, z) for z in ker[0]])
            wt.append(mu)
        Vinv = _inverse_exact(V)
        bar = [wt.index(tuple(-m for m in w)) for w in wt]
        A0 = [next(A for A in range(n) if V[k][A] != Z0) for k in range(n)]
        D[kind] = [_matmul_exact(_matmul_exact(V, Rg), Vinv) for Rg in R]
        out[kind] = {"V": V, "Vinv": Vinv, "wt": wt, "bar": bar,
                     "conj": [zmul(zconj(V[k][A0[bar[k]]]), zinv(V[bar[k]][A0[bar[k]]])) for k in range(n)],
                     "pair": [zsum(zmul(V[k][A], V[bar[k]][A]) for A in range(n)) for k in range(n)],
                     "norm": [sum((a * a + b * b for a, b in row), Fr(0)) for row in V]}
    for name, sign in (("raise", 1), ("lower", -1)):
        maps = []
        for alpha in SIMPLE:
            step = tuple(sign * s for s in alpha)
            eqs = [[Dg[k][l] for Dg in D[kind]] for kind in ("x", "f")
                   for k, l in product(range(len(out[kind]["wt"])), repeat=2)
                   if tuple(b - a for a, b in zip(out[kind]["wt"][k], out[kind]["wt"][l])) != step]
            y = _kernel_exact(eqs, len(gens))
            if len(y) != 1:
                raise RuntimeError(f"root vector {step}: {len(y)} solutions")
            E = {}
            for kind in ("x", "p", "f"):
                E[kind], n = {}, len(out[kind]["wt"])
                for k, l in product(range(n), repeat=2):
                    z = zsum(zmul(c, Dg[k][l]) for c, Dg in zip(y[0], D[kind]) if c != Z0)
                    if z != Z0:
                        if k in E[kind]:
                            raise RuntimeError("root vector maps a weight variable to several")
                        E[kind][k] = (l, z)
            maps.append(E)
        out[name] = maps
    return out


# ---- polynomials in weight coordinates: {(xs, ps, fs): Q(i)}; xs, ps sorted, fs strictly increasing
def perm_sign(seq) -> int:
    return (-1) ** sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq)) if seq[a] > seq[b])


def derive(poly: dict, E: dict) -> dict:
    """A root vector (kind -> {k: (l, c)}) acting as a derivation on a weight-coordinate polynomial."""
    out = defaultdict(lambda: Z0)
    ex, ep, ef = E["x"], E["p"], E["f"]
    for (xs, ps, fs), c in poly.items():
        for i, k in enumerate(xs):
            if k in ex:
                l, e = ex[k]
                key = (tuple(sorted(xs[:i] + xs[i + 1:] + (l,))), ps, fs)
                out[key] = zadd(out[key], zmul(c, e))
        for i, k in enumerate(ps):
            if k in ep:
                l, e = ep[k]
                key = (xs, tuple(sorted(ps[:i] + ps[i + 1:] + (l,))), fs)
                out[key] = zadd(out[key], zmul(c, e))
        for i, k in enumerate(fs):
            if k in ef:
                l, e = ef[k]
                if l in fs:
                    continue
                new = fs[:i] + (l,) + fs[i + 1:]
                s = perm_sign(new)
                key = (xs, ps, tuple(sorted(new)))
                out[key] = zadd(out[key], zmul(c, (e[0] * s, e[1] * s)))
    return {k: v for k, v in out.items() if v != Z0}


@lru_cache(maxsize=None)
def _parts(kind: str, n: int) -> dict:
    """Monomials of one kind and degree n, grouped by weight."""
    wt = action()[kind]["wt"]
    gen = combinations(range(NF), n) if kind == "f" else combinations_with_replacement(range(NX), n)
    out = defaultdict(list)
    for m in gen:
        out[tuple(sum(wt[k][i] for k in m) for i in range(5))].append(m)
    return dict(out)


def monomials(grade: tuple, lam: tuple) -> list:
    """All weight-coordinate monomials of grade (#x, #p, #theta) and weight lam."""
    a, b, c = grade
    out = []
    fparts = _parts("f", c)
    for wx, xs in _parts("x", a).items():
        for wp, ps in _parts("p", b).items():
            fs = fparts.get(tuple(l - u - v for l, u, v in zip(lam, wx, wp)), ())
            out += [(x, p, f) for x in xs for p in ps for f in fs]
    return out


def grade_level(grade: tuple) -> int:
    """Level of the monomials of a grade (#x, #p, #theta): twice their scaling dimension."""
    return 2 * grade[0] + 4 * grade[1] + 3 * grade[2]


def grades(level: int) -> list:
    return [(a, b, c) for a in range(level // 2 + 1) for b in range(level // 4 + 1) for c in range(level // 3 + 1)
            if grade_level((a, b, c)) <= level]


@lru_cache(maxsize=None)
def multiplicities(level: int) -> dict:
    """{(grade, lam): multiplicity} of every irrep (dominant lam) in every grade of level <= level."""
    out = {}
    for a, b, c in grades(level):
        ch = defaultdict(int)
        for wx, xs in _parts("x", a).items():
            for wp, ps in _parts("p", b).items():
                for wf, fs in _parts("f", c).items():
                    ch[tuple(u + v + w for u, v, w in zip(wx, wp, wf))] += len(xs) * len(ps) * len(fs)
        for alpha in ROOTS:                             # Weyl: times prod (1 - z^-alpha)
            new = defaultdict(int, ch)
            for k, v in ch.items():
                new[tuple(x - y for x, y in zip(k, alpha))] -= v
            ch = {k: v for k, v in new.items() if v}
        for (cw, *s), v in ch.items():
            if cw >= 0 and s[0] >= s[1] >= s[2] >= s[3] >= 0 and len({x % 2 for x in s}) == 1 and v > 0:
                out[((a, b, c), (cw, *s))] = v
    return out


def dimension(lam: tuple) -> int:
    """Dimension of the irrep with doubled highest weight lam = (2j; Spin(9) orthonormal x 2)."""
    rho = [Fr(7, 2), Fr(5, 2), Fr(3, 2), Fr(1, 2)]
    l = [Fr(x, 2) + r for x, r in zip(lam[1:], rho)]
    f = lambda v: prod((v[i] - v[j]) * (v[i] + v[j]) for i, j in combinations(range(4), 2)) * prod(v)
    return (lam[0] + 1) * int(f(l) / f(rho))


# ---- exact kernels by elimination modulo primes p = 1 mod 8 and reconstruction over Q(i)
def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@lru_cache(maxsize=None)
def primes_i(count: int, below: int = 1 << 31) -> tuple:
    """Primes p = 1 mod 8 below the bound (i and sqrt2 exist mod p), each with a square root of -1."""
    out, n = [], below - 1
    while len(out) < count:
        if n % 8 == 1 and _is_prime(n):
            g = next(g for g in range(2, n) if pow(g, (n - 1) // 2, n) == n - 1)
            out.append((n, pow(g, (n - 1) // 4, n)))
        n -= 2
    return tuple(out)


def zmod(z, p: int, u: int) -> int:
    """re + u im modulo p for a pair of Fractions: u is the image of i (or of sqrt2, for Q(sqrt2))."""
    re, im = z
    return (re.numerator * pow(re.denominator, -1, p) + u * im.numerator * pow(im.denominator, -1, p)) % p


def matmul_mod(A: np.ndarray, B, p: int) -> np.ndarray:
    """A @ B mod p (A dense or sparse) for entries in [0, p), p < 2^31, inner dimension < 2^17."""
    return ((A @ (B >> 16)) % p * 65536 + A @ (B & 0xFFFF)) % p


def rref_mod(A: np.ndarray, p: int):
    """Reduced row echelon form modulo p (int64 entries in [0, p), p < 2^31): (rows, pivot columns)."""
    A = A.copy()
    piv, r = [], 0
    for c in range(A.shape[1]):
        if r == A.shape[0]:
            break
        nz = np.flatnonzero(A[r:, c])
        if not nz.size:
            continue
        k = r + int(nz[0])
        A[[r, k]] = A[[k, r]]
        A[r] = A[r] * pow(int(A[r, c]), -1, p) % p
        f = A[:, c].copy()
        f[r] = 0
        rows = np.flatnonzero(f)
        if rows.size:
            A[rows] = (A[rows] - f[rows, None] * A[r][None, :] % p) % p
        piv.append(c)
        r += 1
    return A[:r], piv


def inverse_mod(A, p: int) -> np.ndarray:
    """Inverse of a square matrix modulo p (p < 2^31), from the reduced form of [A | 1]."""
    n = len(A)
    red, piv = rref_mod(np.concatenate([np.asarray(A, dtype=np.int64) % p, np.eye(n, dtype=np.int64)], axis=1), p)
    if piv != list(range(n)):
        raise ValueError(f"singular matrix modulo {p}")
    return red[:, n:]


def ratrec(x: int, m: int):
    """Rational reconstruction of x mod m with |num|, den <= sqrt(m/2); None if none exists."""
    if x == 0:
        return Fr(0)
    bound = int((m // 2) ** 0.5)
    r0, r1, s0, s1 = m, x, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return Fr(r1, s1)


def exact_kernel(matrix_mod, ncols: int, real: bool, seed: int = 7) -> list:
    """Kernel of a matrix over Q(i) (over Q if real), as its reduced echelon basis.

    matrix_mod(p, ip) returns the matrix modulo p (int64 array or sparse, any number of rows).  Tall
    matrices are first compressed by random +-1 combinations of rows (the kernel is checked on the
    original).  More primes are used until the reconstruction agrees with a held-out prime."""
    rng = np.random.default_rng(seed)
    pis = primes_i(12)
    images, free = [], None
    for t, (p, ip) in enumerate(pis):
        per = []
        for s in ((ip,) if real else (ip, p - ip)):
            A = matrix_mod(p, s)
            if A.shape[0] > ncols + 8:
                C = sparse.csr_matrix(rng.integers(-1, 2, size=(ncols + 8, A.shape[0])))
                B = C @ A
                B = (B.toarray() if sparse.issparse(B) else np.asarray(B)) % p
            else:
                B = (A.toarray() if sparse.issparse(A) else np.asarray(A)) % p
            red, piv = rref_mod(B, p)
            pset = set(piv)
            f = [c for c in range(ncols) if c not in pset]
            if free is None:
                free = f
            elif f != free:
                raise RuntimeError("kernel structure differs between primes")
            K = np.zeros((len(f), ncols), dtype=np.int64)
            for i, c in enumerate(f):
                K[i, c] = 1
                K[i, piv] = (-red[:, c]) % p
            if f and matmul_mod(A, K.T, p).any():
                raise RuntimeError("compressed kernel is not a kernel")
            per.append(K)
        images.append(per)
        if not free:
            return []
        if t >= 2:
            out = _reconstruct(images[:t], pis[:t], real)
            if out is not None and _agrees(out, images[t], pis[t], real):
                return out
    raise RuntimeError("rational reconstruction of a kernel failed with 12 primes")


def _reconstruct(images, pis, real):
    m, res = 1, None
    for (p, ip), per in zip(pis, images):
        if real:
            parts = (per[0], np.zeros_like(per[0]))
        else:
            Kp, Km = per
            inv2 = pow(2, -1, p)
            parts = ((Kp + Km) % p * inv2 % p, (Kp - Km) % p * inv2 % p * pow(ip, -1, p) % p)
        parts = [x.astype(object) for x in parts]
        res = parts if res is None else [r + m * ((x - r) * pow(m, -1, p) % p) for r, x in zip(res, parts)]
        m *= p
    out = []
    for rre, rim in zip(*res):
        v = []
        for a, b in zip(rre, rim):
            ra, rb = ratrec(int(a) % m, m), ratrec(int(b) % m, m)
            if ra is None or rb is None:
                return None
            v.append((ra, rb))
        out.append(v)
    return out


def _agrees(vecs, per, pr, real) -> bool:
    p, ip = pr
    try:
        return all(zmod(z, p, s) == int(K[i, j]) for s, K in zip((ip,) if real else (ip, p - ip), per)
                   for i, v in enumerate(vecs) for j, z in enumerate(v))
    except ValueError:                                  # a denominator divisible by p
        return False


def _sparse_mod(rows: list, ncols: int, p: int, ip: int):
    """Sparse matrix modulo p from rows given as {column: Q(i)}."""
    r, c, v = [], [], []
    for i, row in enumerate(rows):
        for j, z in row.items():
            r.append(i)
            c.append(j)
            v.append(zmod(z, p, ip))
    return sparse.csr_matrix((np.array(v, dtype=np.int64), (r, c)), shape=(len(rows), ncols))


def vec_mod(vecs: list, p: int, ip: int) -> np.ndarray:
    return np.array([[zmod(z, p, ip) for z in v] for v in vecs], dtype=np.int64).reshape(len(vecs), -1)


# ---- highest-weight vectors and the valley growth, per grade and weight
def highest_weight_vectors(mons: list) -> list:
    rows = []
    for E in action()["raise"]:
        img = defaultdict(dict)
        for i, m in enumerate(mons):
            for t, c in derive({m: Z1}, E).items():
                img[t][i] = c
        rows += list(img.values())
    real = all(z[1] == 0 for row in rows for z in row.values())
    return exact_kernel(lambda p, ip: _sparse_mod(rows, len(mons), p, ip), len(mons), real)


def _orthogonal(n: int, rng, p: int) -> np.ndarray:
    """A random orthogonal matrix modulo p (product of rational Householder reflections)."""
    Q = np.eye(n, dtype=object)
    for _ in range(4):
        v = [int(x) for x in rng.integers(-9, 10, size=n)]
        vv = sum(x * x for x in v) % p
        if vv == 0:
            continue
        inv = pow(vv, -1, p)
        Q = Q.dot(np.array([[((i == j) - 2 * v[i] * v[j] * inv) % p for j in range(n)] for i in range(n)],
                           dtype=object)) % p
    return Q


def valley_map(mons: list, grade: tuple, p: int, ip: int, points: int, rng, dmin: int = TAIL_ROW) -> np.ndarray:
    """(points x degrees, len(mons)) modulo p: the coefficients of mu^d (d >= dmin) of each monomial at
    random valley points: x = mu^2 valley + mu^-1 rest, p = mu^-2 slow + mu fast in a random frame
    (valley: color 0 along axis 0; slow: color 0 or along the valley axis), fermion words hashed."""
    a, b, _ = grade
    shift, top = a + 2 * b, 3 * (a + b)                # mu^shift x monomial: polynomial of degree <= top
    keep = [d for d in range(top + 1) if d - shift >= dmin]
    Vx = np.array([[zmod(z, p, ip) for z in row] for row in action()["x"]["V"]], dtype=object)
    ts = range(1, top + 2)
    Vinv = inverse_mod([[pow(t, d, p) for d in range(top + 1)] for t in ts], p).astype(object)
    xi = np.array([m[0] for m in mons], dtype=np.int64).reshape(len(mons), a)
    pi = np.array([m[1] for m in mons], dtype=np.int64).reshape(len(mons), b)
    slow = np.array([B < 9 or B % 9 == 0 for B in range(NX)])
    out = []
    for _ in range(points):
        W = np.kron(_orthogonal(3, rng, p), _orthogonal(9, rng, p)) % p
        r = [int(v) for v in rng.integers(1, p, size=3 * NX)]
        comps = (np.array([r[0]] + [0] * (NX - 1), dtype=object),
                 np.array([0] + r[1:NX], dtype=object),
                 np.array([r[NX + B] if slow[B] else 0 for B in range(NX)], dtype=object),
                 np.array([0 if slow[B] else r[2 * NX + B] for B in range(NX)], dtype=object))
        al, be, ga, de = (Vx.dot(W.dot(v) % p) % p for v in comps)
        salt = int(rng.integers(1, 1 << 30))
        h = np.array([hash((m[2], salt)) % (p - 1) + 1 for m in mons], dtype=np.int64)
        vals = np.zeros((len(ts), len(mons)), dtype=object)
        for j, t in enumerate(ts):
            X = np.array([(al[k] * t ** 3 + be[k]) % p for k in range(NX)], dtype=np.int64)     # t x(t)
            P = np.array([(ga[k] + de[k] * t ** 3) % p for k in range(NX)], dtype=np.int64)     # t^2 p(t)
            v = h.copy()
            for col in range(a):
                v = v * X[xi[:, col]] % p
            for col in range(b):
                v = v * P[pi[:, col]] % p
            vals[j] = v
        out.append((Vinv.dot(vals) % p)[keep].astype(np.int64))
    return np.vstack(out)


def valley_degrees(mons: list, grade: tuple, vecs: list, seed: int) -> list:
    """Valley degree (2e) of each vector over the monomials of one grade, read off at two random valley points."""
    p, ip = primes_i(1)[0]
    T = valley_map(mons, grade, p, ip, 2, np.random.default_rng(seed), dmin=-10 ** 6)
    out = []
    for v in vecs:
        d = matmul_mod(T, vec_mod([v], p, ip).T, p)[:, 0].reshape(2, -1)
        out.append(max(j for j in range(d.shape[1]) if d[:, j].any()) - (grade[0] + 2 * grade[1]))
    return out


@dataclass
class Sector:
    """The multiplets of irrep lam in one grade: monomials, highest-weight vectors, rows and their growth."""
    grade: tuple
    lam: tuple
    mons: list
    hw: list                # exact vectors over mons
    rows: list              # exact vectors over mons: the finite-norm multiplets
    tails: list             # valley degree 2e of each row

    def poly(self, i: int) -> dict:
        return {m: z for m, z in zip(self.mons, self.rows[i]) if z != Z0}


def sector(grade: tuple, lam: tuple, mult: int) -> Sector:
    """Highest-weight vectors of irrep lam in a grade (checked against the multiplicity) and the rows."""
    mons = monomials(grade, lam)
    hw = highest_weight_vectors(mons)
    if len(hw) != mult:
        raise RuntimeError(f"{grade} {lam}: {len(hw)} highest-weight vectors, the character says {mult}")
    a, b, _ = grade
    ndeg = max(0, 2 * a + b - TAIL_ROW + 1)
    rows = hw
    if ndeg:
        points = -(-mult // ndeg) + 3

        def tail_mod(p, ip):
            T = valley_map(mons, grade, p, ip, points, np.random.default_rng(p))
            return matmul_mod(T, vec_mod(hw, p, ip).T, p)
        combos = exact_kernel(tail_mod, len(hw), False)
        rows = [[zsum(zmul(c, v[j]) for c, v in zip(cb, hw) if c != Z0) for j in range(len(mons))] for cb in combos]
    return Sector(grade, lam, mons, hw, rows, valley_degrees(mons, grade, rows, 99))


# ---- weight coordinates <-> Cartesian Weyl symbols (exact)
def _expand(terms: dict, kind: str, k: int, matrix) -> dict:
    """Multiply the partial terms {(xs, ps, fs): z} on the right by sum_A matrix[k][A] variable_A."""
    new = defaultdict(lambda: Z0)
    for (bx, bp, bf), v in terms.items():
        for A, w in enumerate(matrix[k]):
            if w == Z0:
                continue
            if kind == "x":
                key = (tuple(sorted(bx + (A,))), bp, bf)
            elif kind == "p":
                key = (bx, tuple(sorted(bp + (A,))), bf)
            elif A in bf:
                continue
            else:
                key = (bx, bp, bf + (A,))
            new[key] = zadd(new[key], zmul(v, w))
    return new


def to_cartesian(poly: dict) -> Poly:
    """A weight-coordinate polynomial as a Poly in the real variables."""
    act = action()
    acc = defaultdict(lambda: Z0)
    for (xs, ps, fs), c in poly.items():
        terms = {((), (), ()): c}
        for kind, idx in (("x", xs), ("p", ps), ("f", fs)):
            for k in idx:
                terms = _expand(terms, kind, k, act[kind]["V"])
        for (bx, bp, bf), v in terms.items():
            n, s = len(bf), perm_sign(bf)
            v = zrot((v[0] * s, v[1] * s), -(n * (n - 1) // 2))   # ordered product = i^(-n(n-1)/2) basis word
            key = (bx, bp, sum(1 << j for j in bf))
            acc[key] = zadd(acc[key], v)
    acc = {k: v for k, v in acc.items() if v != Z0}
    den = lcm(*(f.denominator for v in acc.values() for f in v)) if acc else 1
    return Poly({k: (int(v[0] * den), 0, int(v[1] * den), 0) for k, v in acc.items()}, den).normal()


def from_cartesian(P: Poly) -> tuple:
    """A Poly as weight-coordinate polynomials (rational part, sqrt2 part)."""
    act = action()
    out = (defaultdict(lambda: Z0), defaultdict(lambda: Z0))
    for (xs, ps, mask), (a, b, c, d) in P.terms.items():
        js = [j for j in range(NF) if mask >> j & 1]
        n = len(js)
        for part, (re, im) in enumerate(((a, c), (b, d))):
            if not re and not im:
                continue
            terms = {((), (), ()): zrot((Fr(re, P.den), Fr(im, P.den)), n * (n - 1) // 2)}
            for kind, idx in (("x", xs), ("p", ps), ("f", js)):
                for A in idx:
                    terms = _expand(terms, kind, A, act[kind]["Vinv"])
            for (bx, bp, bf), v in terms.items():
                s = perm_sign(bf)
                key = (bx, bp, tuple(sorted(bf)))
                out[part][key] = zadd(out[part][key], (v[0] * s, v[1] * s))
    return tuple({k: v for k, v in o.items() if v != Z0} for o in out)
