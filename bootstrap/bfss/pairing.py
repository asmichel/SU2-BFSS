"""Invariant pairings of multiplets and their projections modulo primes.

A multiplet is given by a highest-weight vector h (a weight-coordinate polynomial, basis.py).  A tree of
simple lowering steps, chosen once per irrep, spans the irrep from any highest-weight vector.  The Fischer
inner products of the tree vectors of one copy (its Shapovalov form) are the same for every copy up to a
factor; Gram-Schmidt within each weight space gives orthogonal combinations o_k with norms D_k, and

    pair(h, h') = sum_k o_k(h)^+ * o_k(h') / D_k          (* the Moyal-Clifford product)

is the invariant pairing of two copies.

Forms are compared through K functionals modulo primes p = 1 mod 8 below 2^20 (i and sqrt2 exist; float64
products of residues are exact): functional j evaluates a symbol at the point (g_v^j) of the weight
coordinates v of x and p and weights a fermion word w by prod_{v in w} g_v^j.  The star product is a finite
sum over contraction channels (x-p pairs, theta pairs), each a pointwise product of derivatives, so
psi_j(pair(h, h')) is a bilinear combination of evaluations of derivatives of components: no product
polynomial is ever expanded.  Fermion signs enter through a channel matrix of signs.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from functools import lru_cache
from itertools import combinations, product
from math import factorial

import numpy as np
from scipy import sparse

from . import basis as B
from .basis import Z0, Z1, perm_sign, zadd, zconj, zmul, zsum
from .kinematics import falling

LIMB = 1 << 20
SLICE = 1 << 26                    # numbers in the power sums of one slice of functionals


@lru_cache(maxsize=None)
def primes(count: int) -> tuple:
    """(p, i, sqrt2) modulo primes p = 1 mod 8 below 2^20."""
    out = []
    for p, ip in B.primes_i(count, LIMB):
        g = next(g for g in range(2, p) if pow(g, (p - 1) // 2, p) == p - 1)
        z = pow(g, (p - 1) // 8, p)
        out.append((p, ip, (z + pow(z, -1, p)) % p))
    return tuple(out)


# ---- exact components
def fischer(f: dict, g: dict):
    """The Fischer inner product <f, g> (antilinear in f; invariant)."""
    act = B.action()
    out = Z0
    for key, b in g.items():
        a = f.get(key)
        if a is None:
            continue
        xs, ps, fs = key
        w = Fr(1)
        for kind, idx in (("x", xs), ("p", ps)):
            for k, n in Counter(idx).items():
                w *= factorial(n) * act[kind]["norm"][k] ** n
        for k in fs:
            w *= act["f"]["norm"][k]
        out = zadd(out, zmul(zconj(a), (b[0] * w, b[1] * w)))
    return out


def _lin(terms) -> dict:
    """sum of coefficient * polynomial."""
    acc = defaultdict(lambda: Z0)
    for c, P in terms:
        if c != Z0:
            for m, z in P.items():
                acc[m] = zadd(acc[m], zmul(c, z))
    return {m: z for m, z in acc.items() if z != Z0}


class Irrep:
    """Lowering tree and orthogonal components of one irrep, from a reference highest-weight vector."""

    def __init__(self, lam: tuple, ref: dict):
        self.lam = lam
        act = B.action()
        p, ip = B.primes_i(1)[0]
        nodes, vecs, wts = [(-1, -1)], [ref], [lam]
        ech = defaultdict(list)                       # weight -> reduced rows modulo p
        _reduce(ech[lam], ref, p, ip)
        frontier, dim = [0], B.dimension(lam)
        while frontier and len(nodes) < dim:
            nxt = []
            for n in frontier:
                for i, alpha in enumerate(B.SIMPLE):
                    v = B.derive(vecs[n], act["lower"][i])
                    w = tuple(a - b for a, b in zip(wts[n], alpha))
                    if v and _reduce(ech[w], v, p, ip):
                        nodes.append((n, i))
                        vecs.append(v)
                        wts.append(w)
                        nxt.append(len(nodes) - 1)
            frontier = nxt
        if len(nodes) != dim:
            raise RuntimeError(f"lowering tree of {lam}: {len(nodes)} vectors, dimension {dim}")
        self.nodes = nodes
        self.comps = []                               # [(D_k, [(node, coefficient)])]
        self.weights = []                             # weight of each component
        groups = defaultdict(list)
        for n, w in enumerate(wts):
            groups[w].append(n)
        for ns in groups.values():
            S = [[fischer(vecs[a], vecs[b]) for b in ns] for a in ns]
            done = []
            for i in range(len(ns)):
                a = [Z1 if c == i else Z0 for c in range(len(ns))]
                for D, ak in done:
                    f = zsum(zmul(zconj(ak[c]), S[c][i]) for c in range(len(ns)) if ak[c] != Z0)
                    f = (-f[0] / D, -f[1] / D)
                    a = [zadd(x, zmul(f, y)) for x, y in zip(a, ak)]
                D = zsum(zmul(zconj(a[c]), zmul(S[c][d], a[d])) for c in range(len(ns)) for d in range(len(ns))
                         if a[c] != Z0 and a[d] != Z0)
                if D[1] != 0 or D[0] <= 0:
                    raise RuntimeError("Fischer form not positive")
                done.append((D[0], a))
            self.comps += [(D, [(n, c) for n, c in zip(ns, a) if c != Z0]) for D, a in done]
            self.weights += [wts[ns[0]]] * len(done)

    def components(self, h: dict) -> list:
        """The orthogonal components o_k(h) of the copy with highest-weight vector h (exact)."""
        lower = B.action()["lower"]
        vecs = [h]
        for parent, i in self.nodes[1:]:
            vecs.append(B.derive(vecs[parent], lower[i]))
        return [_lin((c, vecs[n]) for n, c in terms) for _, terms in self.comps]

    def raising(self, comps: list) -> list:
        """Matrices of the simple raising operators on the components of a copy: per root {(k', k): z}
        with E_i o_k = sum_k' z o_k' (Fischer-orthogonal components; the same for every copy)."""
        act = B.action()
        by_w = defaultdict(list)
        for k, w in enumerate(self.weights):
            by_w[w].append(k)
        out = []
        for i, E in enumerate(act["raise"]):
            Mi = {}
            for k, o in enumerate(comps):
                v = B.derive(o, E)
                w = tuple(a + b for a, b in zip(self.weights[k], B.SIMPLE[i]))
                for k2 in by_w.get(w, []) if v else []:
                    c = fischer(comps[k2], v)
                    if c != Z0:
                        D = self.comps[k2][0]
                        Mi[(k2, k)] = (c[0] / D, c[1] / D)
            out.append(Mi)
        return out


def tensor_highest(A: Irrep, RA: list, Bi: Irrep, RB: list, lam: tuple) -> list:
    """Highest-weight vectors of irrep lam in the tensor product of two irreps (raising matrices RA, RB):
    exact coefficient dicts {(k, d): z} on the products of components."""
    pairs = [(k, d) for k, wa in enumerate(A.weights) for d, wb in enumerate(Bi.weights)
             if tuple(x + y for x, y in zip(wa, wb)) == lam]
    if not pairs:
        return []
    col = {pr: n for n, pr in enumerate(pairs)}
    eqs = defaultdict(lambda: [Z0] * len(pairs))
    for i in range(len(B.SIMPLE)):
        for (k2, k), z in RA[i].items():
            for (kk, d), n in col.items():
                if kk == k:
                    eqs[(i, k2, d)][n] = zadd(eqs[(i, k2, d)][n], z)
        for (d2, d), z in RB[i].items():
            for (k, dd), n in col.items():
                if dd == d:
                    eqs[(i, k, d2, "b")][n] = zadd(eqs[(i, k, d2, "b")][n], z)
    rows = {}
    for key, row in eqs.items():                      # (i, k', d') targets from both terms coincide
        tgt = key[:3]
        rows[tgt] = [zadd(a, b) for a, b in zip(rows.get(tgt, [Z0] * len(pairs)), row)]
    return [{pairs[n]: z for n, z in enumerate(v) if z != Z0} for v in B._kernel_exact(list(rows.values()), len(pairs))]


def _reduce(rows: list, v: dict, p: int, ip: int) -> bool:
    """Add v (modulo p) to reduced echelon rows {monomial: residue} if independent of them."""
    r = {m: B.zmod(z, p, ip) for m, z in v.items()}
    for row in rows:
        piv = row["_piv"]
        c = r.get(piv, 0)
        if c:
            for m, x in row.items():
                if m != "_piv":
                    r[m] = (r.get(m, 0) - c * x) % p
    r = {m: x for m, x in r.items() if x}
    if not r:
        return False
    piv = min(r)
    inv = pow(r[piv], -1, p)
    rows.append({**{m: x * inv % p for m, x in r.items()}, "_piv": piv})
    return True


def dagger(poly: dict) -> dict:
    """Hermitian conjugate of a weight-coordinate symbol."""
    act = B.action()
    cx, cp, cf = act["x"]["conj"], act["p"]["conj"], act["f"]["conj"]
    bx, bp, bf = act["x"]["bar"], act["p"]["bar"], act["f"]["bar"]
    out = {}
    for (xs, ps, fs), c in poly.items():
        z = zconj(c)
        for k in xs:
            z = zmul(z, cx[k])
        for k in ps:
            z = zmul(z, cp[k])
        for k in fs:
            z = zmul(z, cf[k])
        seq = [bf[k] for k in reversed(fs)]           # (w_1 ... w_n)^+ = w_n^+ ... w_1^+
        s = perm_sign(seq)
        out[(tuple(sorted(bx[k] for k in xs)), tuple(sorted(bp[k] for k in ps)), tuple(sorted(seq)))] = \
            (z[0] * s, z[1] * s)
    return out


# ---- contraction channels
def _submultisets(seq: tuple, maxsize: int):
    """(sub, rest, falling factor) over sub-multisets of a sorted tuple, |sub| <= maxsize."""
    cnt = sorted(Counter(seq).items())
    for ks in product(*(range(min(n, maxsize) + 1) for _, n in cnt)):
        if sum(ks) > maxsize:
            continue
        sub, rest, fall = [], [], 1
        for (k, n), c in zip(cnt, ks):
            sub += [k] * c
            rest += [k] * (n - c)
            fall *= falling(n, c)
        yield tuple(sub), tuple(rest), fall


def _subsets(fs: tuple):
    for r in range(len(fs) + 1):
        for A in combinations(fs, r):
            yield A, tuple(k for k in fs if k not in A)


@lru_cache(maxsize=1 << 20)
def contraction_sign(u: tuple, v: tuple, A: tuple) -> int:
    """Sign of the theta contraction A (subset of u, paired with bar A in v) in u * v; 0 if the remaining
    words overlap.  (u d_a1 ... d_an)(d_bn ... d_b1 v), right derivatives on u and left ones on v."""
    bar = B.action()["f"]["bar"]
    s, cu, cv = 1, list(u), list(v)
    for a in A:
        i = cu.index(a)
        s *= (-1) ** (len(cu) - 1 - i)
        del cu[i]
        j = cv.index(bar[a])
        s *= (-1) ** j
        del cv[j]
    seq = cu + cv
    if len(set(seq)) < len(seq):
        return 0
    return s * perm_sign(seq)


def _kappa(S: tuple, T: tuple, A: tuple):
    """Exact coefficient of the channel: (i/2)^|S| (-i/2)^|T| prod pair / multiplicities! (1/2)^|A| prod pair_f."""
    act = B.action()
    z = Z1
    for k in S:
        z = zmul(z, (Fr(0), Fr(1, 2)))
        z = zmul(z, act["x"]["pair"][k])
    for l in T:
        z = zmul(z, (Fr(0), Fr(-1, 2)))
        z = zmul(z, act["p"]["pair"][l])
    for a in A:
        z = zmul(z, (Fr(1, 2) * act["f"]["pair"][a][0], Fr(1, 2) * act["f"]["pair"][a][1]))
    m = 1
    for _, n in Counter(S).items():
        m *= factorial(n)
    for _, n in Counter(T).items():
        m *= factorial(n)
    return (z[0] / m, z[1] / m)


def star(f: dict, g: dict) -> dict:
    """The Moyal-Clifford product of weight-coordinate symbols (exact; for checks)."""
    bx, bp, bf = (B.action()[k]["bar"] for k in "xpf")
    acc = defaultdict(lambda: Z0)
    for (xs, ps, fs), a in f.items():
        for (ys, qs, gs), b in g.items():
            for T, ps_r, ft in _submultisets(ps, len(ys)):
                for S, xs_r, fsx in _submultisets(xs, len(qs)):
                    qs_r = _remove(qs, tuple(sorted(bp[k] for k in S)))
                    ys_r = _remove(ys, tuple(sorted(bx[l] for l in T)))
                    if qs_r is None or ys_r is None:
                        continue
                    fq = _falling(qs, tuple(sorted(bp[k] for k in S)))
                    fy = _falling(ys, tuple(sorted(bx[l] for l in T)))
                    for A, fs_r in _subsets(fs):
                        if not all(bf[k] in gs for k in A):
                            continue
                        sg = contraction_sign(fs, gs, A)
                        if not sg:
                            continue
                        gs_r = tuple(k for k in gs if k not in {bf[a_] for a_ in A})
                        seq = fs_r + gs_r
                        z = zmul(zmul(a, b), _kappa(S, T, A))
                        n = ft * fsx * fq * fy * sg
                        key = (tuple(sorted(xs_r + ys_r)), tuple(sorted(ps_r + qs_r)), tuple(sorted(seq)))
                        acc[key] = zadd(acc[key], (z[0] * n, z[1] * n))
    return {m: z for m, z in acc.items() if z != Z0}


def _remove(seq: tuple, sub: tuple):
    rest = list(seq)
    for k in sub:
        if k not in rest:
            return None
        rest.remove(k)
    return tuple(rest)


def _falling(seq: tuple, sub: tuple) -> int:
    c = Counter(seq)
    out = 1
    for k, n in Counter(sub).items():
        out *= falling(c[k], n)
    return out


# ---- projections modulo a prime
class Functionals:
    """The K evaluation functionals modulo one prime."""

    def __init__(self, prime: tuple, K: int, seed: int):
        self.p, self.ip, _ = prime
        self.K = K
        rng = np.random.default_rng(seed + self.p)
        # values of the weight coordinates: x (0..26), p (27..53), theta (54..101); 102 pads
        self.vals = np.append(rng.integers(1, self.p, 2 * B.NX + B.NF), 1)

    def z(self, c) -> int:
        return B.zmod(c, self.p, self.ip)

    def power_sums(self, key: np.ndarray, coef: np.ndarray, node: np.ndarray, nkeys: int, js: range) -> np.ndarray:
        """(nkeys, len(js)) float64: sum over items with each key of coef * node^j, j in js, modulo p."""
        p = self.p
        out = np.zeros((nkeys, len(js)))
        step = max(1, (1 << 24) // len(js))
        for lo in range(0, len(key), step):
            nd, ky = node[lo:lo + step], key[lo:lo + step]
            pw = np.empty((len(js), len(nd)), dtype=np.int64)
            pw[0] = coef[lo:lo + step] * _powmod(nd, js.start, p) % p
            for j in range(1, len(js)):
                pw[j] = pw[j - 1] * nd % p
            S = sparse.csr_matrix((np.ones(len(ky)), (ky, np.arange(len(ky)))), shape=(nkeys, len(ky)))
            out = np.fmod(out + S @ pw.T.astype(np.float64), p)   # sums of < 2^33 residues are exact
        return out


def _powmod(x: np.ndarray, e: int, p: int) -> np.ndarray:
    out, b = np.ones_like(x), x % p
    while e:
        if e & 1:
            out = out * b % p
        b = b * b % p
        e >>= 1
    return out


class _Items:
    """Channel items of components in one role, independent of the prime.  Per item: its slot (component,
    channel, fermion word; shared by all copies), its copy, the term (exact coefficient stored once), an
    integer multiplicity, the channel coefficient (stored once) and the variables of the remaining monomial."""

    def __init__(self):
        self.slots, self.terms = {}, []
        self.slot, self.copy, self.term, self.fac, self.chan = (array("q") for _ in range(5))
        self.vars, self.width = array("b"), array("b")          # remaining variables, packed

    def add(self, slot, copy: int, term: int, fac: int, chan: int, rest):
        self.slot.append(self.slots.setdefault(slot, len(self.slots)))
        self.copy.append(copy)
        self.term.append(term)
        self.fac.append(fac)
        self.chan.append(chan)
        self.vars.extend(rest)
        self.width.append(len(rest))

    def finish(self, ncopies: int) -> None:
        """Compact keys: the (slot, copy) pairs that occur."""
        comb = np.frombuffer(self.slot, dtype=np.int64) * ncopies + np.frombuffer(self.copy, dtype=np.int64)
        uniq, self.key = np.unique(comb, return_inverse=True)
        self.key_slot, self.key_copy = uniq // ncopies, uniq % ncopies

    def prepare(self, fun: Functionals, chan_coef: list) -> tuple:
        """Keys, coefficients and nodes of the items modulo the prime (chan_coef: exact, per channel)."""
        p = fun.p
        zt = np.array([fun.z(z) for z in self.terms], dtype=np.int64)
        zc = np.array([fun.z(z) for z in chan_coef], dtype=np.int64)
        coef = zt[np.frombuffer(self.term, dtype=np.int64)] * (np.frombuffer(self.fac, dtype=np.int64) % p) % p \
            * zc[np.frombuffer(self.chan, dtype=np.int64)] % p
        width = np.frombuffer(self.width, dtype=np.int8).astype(np.int64)
        flat = np.frombuffer(self.vars, dtype=np.int8).astype(np.int64)
        node, vals = np.ones(len(width), dtype=np.int64), fun.vals[flat]
        first = np.cumsum(width) - width                             # offset of each item's first variable
        for c in range(int(width.max(initial=0))):                   # times the c-th variable of each item
            sel = width > c
            node[sel] = node[sel] * vals[first[sel] + c] % p
        return self.key, coef, node, len(self.key_slot)


def _rest(xs, ps, fs) -> tuple:
    return tuple(xs) + tuple(B.NX + k for k in ps) + tuple(2 * B.NX + k for k in fs)


class Pairing:
    """psi_j(pair(h_a, h'_b)) for left copies a and right copies b (left[a], right[b]: exact components,
    the left ones conjugated).  The channel structure is built once; evaluate() works modulo a prime.

    Items are summed into vectors per slot (component, channel, word) and copy.  A channel couples left and
    right words through a sign matrix that does not depend on the copies, so all channels form one sparse
    matrix E (left slots x right slots), and G_j = L_j^T (E R_j) is a batched product over the copies."""

    def __init__(self, left: list, right: list, irrep: Irrep):
        bx, bp, bf = (B.action()[k]["bar"] for k in "xpf")
        self.shape = (len(left), len(right))
        self.L, self.R = _Items(), _Items()
        lchan = {}                                 # left: (component, channel) -> id
        self.lcoef = []                            # exact kappa / D_k per left channel id
        for k, (D, _) in enumerate(irrep.comps):
            Ls, Rs = [c[k] for c in left], [c[k] for c in right]
            sets = _sets(Ls, Rs)
            for c, P in enumerate(Ls):
                for (xs, ps, fs), z in P.items():
                    t = len(self.L.terms)
                    self.L.terms.append(z)
                    Ss = [(S, xr, fx) for S, xr, fx in _submultisets(xs, sets["maxR"]) if S in sets["PR"]]
                    As = [(A, fr) for A, fr in _subsets(fs) if A in sets["FR"]]
                    for T, prest, ft in _submultisets(ps, 99):
                        for S, xrest, fx in Ss:
                            for A, frest in As:
                                ch = (A, S, T)
                                cid = lchan.get((k, ch))
                                if cid is None:
                                    cid = lchan[(k, ch)] = len(self.lcoef)
                                    kz = _kappa(S, T, A)
                                    self.lcoef.append((kz[0] / D, kz[1] / D))
                                self.L.add((k, ch, fs), c, t, ft * fx, cid, _rest(xrest, prest, frest))
            for c, P in enumerate(Rs):
                for (xs, ps, fs), z in P.items():
                    t = len(self.R.terms)
                    self.R.terms.append(z)
                    Ts = [(tuple(sorted(bx[k_] for k_ in Tp)), xr, ft) for Tp, xr, ft in _submultisets(xs, sets["maxL"])]
                    Ts = [x for x in Ts if x[0] in sets["PL"]]
                    As = [(tuple(sorted(bf[k_] for k_ in Ap)), fr) for Ap, fr in _subsets(fs)]
                    As = [x for x in As if x[0] in sets["FL"]]
                    for Sp, prest, fs_ in _submultisets(ps, 99):
                        S = tuple(sorted(bp[k_] for k_ in Sp))
                        for T, xrest, ft in Ts:
                            for A, frest in As:
                                self.R.add((k, (A, S, T), fs), c, t, ft * fs_, 0, _rest(xrest, prest, frest))
        rby = defaultdict(list)
        for (k, ch, v), j in self.R.slots.items():
            rby[(k, ch)].append((v, j))
        rows, cols, vals = array("q"), array("q"), array("d")
        for (k, ch, u), i in self.L.slots.items():
            for v, j in rby.get((k, ch), ()):
                sg = contraction_sign(u, v, ch[0])
                if sg:
                    rows.append(i)
                    cols.append(j)
                    vals.append(sg)
        self.E = sparse.csr_matrix((np.frombuffer(vals), (np.frombuffer(rows, dtype=np.int64), np.frombuffer(cols, dtype=np.int64))),
                                   shape=(len(self.L.slots), len(self.R.slots)))
        self.L.finish(len(left))
        self.R.finish(len(right))
        self.Sa = sparse.csr_matrix((np.ones(len(self.L.key_copy)), (self.L.key_copy, np.arange(len(self.L.key_copy)))),
                                    shape=(len(left), len(self.L.key_copy)))

    def evaluate(self, fun: Functionals, js: range = None) -> np.ndarray:
        """G[a, b, j] = sum over left keys (u, a) of L[(u, a), j] (E R_j)[u, b] modulo p, for the functionals
        j in js (default all), taken in slices so that the dense arrays stay below SLICE numbers."""
        p = fun.p
        js = range(fun.K) if js is None else js
        (na, nb), nl, nr = self.shape, len(self.L.slots), len(self.R.slots)
        Li, Ri = self.L.prepare(fun, self.lcoef), self.R.prepare(fun, [Z1])
        G = np.zeros(self.shape + (len(js),))
        if not self.E.nnz:
            return G.astype(np.int64)
        width = max(1, min(len(js), SLICE // max(nr * nb, nl * nb, Li[3] * nb, 1)))
        for j0 in range(js.start, js.stop, width):
            sl = range(j0, min(js.stop, j0 + width))
            w = len(sl)
            Lv, Rc = fun.power_sums(*Li, sl), fun.power_sums(*Ri, sl)
            Rv = np.zeros((nr, nb, w))
            Rv[self.R.key_slot, self.R.key_copy] = Rc
            Y = np.fmod(self.E @ Rv.reshape(nr, nb * w), p).reshape(nl, nb, w)   # sums of < 2^33 residues: exact
            Z = np.fmod(Y[self.L.key_slot] * Lv[:, None, :], p)                 # (left keys, nb, w)
            G[:, :, sl.start - js.start:sl.stop - js.start] = np.fmod(self.Sa @ Z.reshape(len(Lv), nb * w), p).reshape(na, nb, w)
        return np.mod(G, p).astype(np.int64)


def _sets(lefts: list, rights: list) -> dict:
    """Which contraction channels can occur between left and right components."""
    bp, bf = B.action()["p"]["bar"], B.action()["f"]["bar"]
    return {"maxL": max((len(m[1]) for P in lefts for m in P), default=0),
            "maxR": max((len(m[1]) for P in rights for m in P), default=0),
            "PL": {T for P in lefts for m in P for T, _, _ in _submultisets(m[1], 99)},
            "PR": {tuple(sorted(bp[k] for k in S)) for P in rights for m in P for S, _, _ in _submultisets(m[1], 99)},
            "FL": {A for P in lefts for m in P for A, _ in _subsets(m[2])},
            "FR": {tuple(sorted(bf[k] for k in A)) for P in rights for m in P for A, _ in _subsets(m[2])}}
