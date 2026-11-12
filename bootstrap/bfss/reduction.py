"""Exact reduction of the moment problem.

Every form (Gram entry, Ward identity, observable) is a real-linear functional of the real moment vector.
Forms are compared through K functionals modulo primes p = 1 mod 8 below 2^20 (pairing.py), for both
images of sqrt2.  Pivots are taken in form order (Ward, identity, observables, entries): Ward pivots are
set to zero, the identity to one, and every other form becomes an affine function of the free pivots z.
Coefficients in Q(sqrt2) come from Chinese remaindering and rational reconstruction and are verified on
held-out primes.  Facial reduction: the common kernel of each block's pencil (rows annihilating every
admissible state, as N|Omega> = 0) is removed.  Floating point serves only exact residue arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from fractions import Fraction
from math import prod
from pathlib import Path

import numpy as np

from . import basis as B
from . import moments as Mo
from . import pairing as Pa
from .problem import Block, Problem

_JOB: dict = {}
SPLIT = 16                         # most tasks a block is split into
SPLIT_RATIO = 8                    # a block is split when its cost exceeds this many typical blocks


# ---- projections of the forms; a task is a block, a set of primes and a range of functionals
def _block_forms(task):
    """{label: (primes, 4, functionals) int32}: parts (Re, Re sqrt2, Im, Im sqrt2) of the forms of a block."""
    lam, sel, js = task
    h, K, seed = _JOB["h"], _JOB["K"], _JOB["seed"]
    ps = [_JOB["primes"][t] for t in sel]
    t0 = time.time()
    rows = h.rows.get(lam, [])
    groups = []                                     # (kind, left copies, right copies, labels)
    if rows:
        groups.append(("gram", [r.poly for r in rows], [r.poly for r in rows]))
    ws = [w for w in h.ward if w[1] == lam]          # <r^+ n>: rows against the null multiplets n
    ns = sorted({w[4] for w in ws})
    if ws:
        groups.append(("ward", [r.poly for r in rows], [part for j in ns for part in h.nulls[j][1]]))
    if lam == Mo.SINGLET:
        groups.append(("obs", [rows[0].poly], [part for o in Mo.OBSERVABLES for part in h.special[o]]))
        groups.append(("P4", [h.special["P2"][0]], [h.special["P2"][0]]))
    ref = min((r.poly for r in rows), key=len) if rows else next(c for _, _, R in groups for c in R if c)
    irr = Pa.Irrep(lam, ref)
    comps = {}

    def cmp(P):
        if id(P) not in comps:
            comps[id(P)] = ([{} for _ in irr.comps] if not P else irr.components(P), P)
        return comps[id(P)][0]
    dag = {}

    def dg(P):
        if id(P) not in dag:
            dag[id(P)] = [Pa.dagger(o) for o in cmp(P)]
        return dag[id(P)]
    pairs = {}
    for kind, L, R in groups:
        pairs[(kind, 0)] = Pa.Pairing([dg(P) for P in L], [cmp(P) for P in R], irr)
        if kind in ("ward", "obs"):
            pairs[(kind, 1)] = Pa.Pairing([dg(P) for P in R], [cmp(P) for P in L], irr)
    ncopies = len(comps)
    comps.clear()                                   # the pairings hold what is needed
    dag.clear()
    out = {}
    for t, prime in enumerate(ps):
        fun = Pa.Functionals(prime, K, seed)
        p, ip = fun.p, fun.ip
        inv2, inv2i = pow(2, -1, p), pow(2 * ip, -1, p)

        def parts(a_r, a_s, b_r, b_s):
            return np.stack([(a_r + b_r) * inv2 % p, (a_s + b_s) * inv2 % p,
                             (a_r - b_r) * inv2i % p, (a_s - b_s) * inv2i % p])
        z = np.zeros(len(js), dtype=np.int64)
        for kind, L, R in groups:
            if kind == "gram":
                G = pairs[(kind, 0)].evaluate(fun, js)
                for i in range(len(rows)):
                    for j in range(i, len(rows)):
                        out.setdefault(("entry", lam, i, j), []).append(parts(G[i, j], z, G[j, i], z))
            elif kind == "ward":
                A = pairs[(kind, 0)].evaluate(fun, js)        # (rows, 2 x nulls)
                Bm = pairs[(kind, 1)].evaluate(fun, js)       # (2 x nulls, rows)
                for _, _, label, i, j in ws:
                    c = 2 * ns.index(j)
                    out.setdefault(("ward", label), []).append(parts(A[i, c], A[i, c + 1], Bm[c, i], Bm[c + 1, i]))
            elif kind == "obs":
                A = pairs[(kind, 0)].evaluate(fun, js)
                Bm = pairs[(kind, 1)].evaluate(fun, js)
                for i, o in enumerate(Mo.OBSERVABLES):
                    out.setdefault(("obs", o), []).append(
                        parts(A[0, 2 * i], A[0, 2 * i + 1], Bm[2 * i, 0], Bm[2 * i + 1, 0]))
            else:
                A = pairs[(kind, 0)].evaluate(fun, js)
                out.setdefault(("obs", "P4"), []).append(parts(A[0, 0], z, A[0, 0], z))
    info = {"event": "block", "family": Mo.family(lam), "dim": B.dimension(lam), "rows": len(rows), "copies": ncopies,
            "seconds": round(time.time() - t0, 1), "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)}
    return task, {k: np.stack(v).astype(np.int32) for k, v in out.items()}, info


def project(h: Mo.Hierarchy, ps, K: int, seed: int, workers: int, log=print) -> dict:
    """{label: (len(ps), 4, K)} for all forms.  Blocks are scheduled by estimated cost; the costliest are
    split over primes and ranges of functionals, so that no task is much longer than a typical block."""
    _JOB.update(h=h, K=K, primes=ps, seed=seed)

    def cost(l):
        rs = h.rows.get(l, [])
        return B.dimension(l) * (1 + sum(len(r.poly) for r in rs)) * (1 + len(rs)) ** 2
    lams = sorted(set(h.rows) | {w[1] for w in h.ward}, key=cost, reverse=True)
    typical = sorted(cost(l) for l in lams)[len(lams) // 2] or 1
    tasks = []
    for l in lams:
        n = max(1, min(SPLIT, round(cost(l) / typical / SPLIT_RATIO)))
        per = -(-n // len(ps))                                   # functional slices per prime
        groups = [list(range(len(ps)))] if n == 1 else [[t] for t in range(len(ps))]
        width = -(-K // (per if n > 1 else 1))
        tasks += [(l, g, range(j0, min(K, j0 + width))) for g in groups for j0 in range(0, K, width)]
    out = {}
    if workers <= 1:
        results = map(_block_forms, tasks)
    else:                      # a worker killed (out of memory) raises BrokenProcessPool instead of hanging
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))
        results = (f.result() for f in as_completed([ex.submit(_block_forms, t) for t in tasks]))
    for i, ((l, sel, js), forms, info) in enumerate(results):
        for k, arr in forms.items():
            if k not in out:
                out[k] = np.zeros((len(ps), 4, K), dtype=np.int32)
            out[k][np.ix_(sel, range(4), js)] = arr
        log(json.dumps({**info, "primes": len(sel), "functionals": [js.start, js.stop], "done": i + 1, "of": len(tasks)}))
    if workers > 1:
        ex.shutdown()
    return out


# ---- linear algebra mod p
def _matmul_mod(A: np.ndarray, B_: np.ndarray, p: int) -> np.ndarray:
    """A @ B mod p for residues of p < 2^20: float64 products, exact over inner chunks of 2^13."""
    out = np.zeros((A.shape[0], B_.shape[1]))
    for c in range(0, A.shape[1], 1 << 13):
        out = np.fmod(out + A[:, c:c + (1 << 13)].astype(np.float64) @ B_[c:c + (1 << 13)].astype(np.float64), p)
    return out.astype(np.int64)


def _pivots(rows: np.ndarray, p: int):
    """Rows independent of all earlier rows, and pivot columns (incremental reduced echelon form)."""
    basis = np.zeros((min(rows.shape), rows.shape[1]), dtype=np.int64)
    pcols, prow = [], []
    for i, v in enumerate(rows):
        n = len(pcols)
        if n:
            v = (v - _matmul_mod(v[pcols][None, :], basis[:n], p)[0]) % p
        nz = np.flatnonzero(v)
        if not nz.size:
            continue
        c = int(nz[0])
        v = v * pow(int(v[c]), -1, p) % p
        if n:
            basis[:n] = (basis[:n] - basis[:n, [c]] * v[None, :] % p) % p
        basis[n] = v
        pcols.append(c)
        prow.append(i)
    return prow, pcols


def code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(root.glob("*.py"))}


# ---- the compiler
MAX_PRIMES = 24


def _images(arrs: list, ps, K: int) -> list:
    """Per prime, the matrices of Re and Im parts of all forms (rows 2f, 2f + 1) for both images of sqrt2,
    from the (F, 4, K) projections."""
    out = []
    for arr, (p, ip, s) in zip(arrs, ps):
        comps = arr[:, :, :K].astype(np.int64)
        out.append([np.stack([(comps[:, 0] + sg * comps[:, 1]) % p, (comps[:, 2] + sg * comps[:, 3]) % p],
                             axis=1).reshape(-1, K) for sg in (s, p - s)])
    return out


def _coefficients(Vs: list, prime, piv: list, need, cols, rng) -> tuple:
    """Coordinates of every needed row in the pivot basis modulo one prime, as a + b sqrt2 parts."""
    p, ip, s = prime
    rank = len(piv)
    later = np.array(piv)[None, :] > np.arange(len(Vs[0]))[:, None]
    C = []
    for Vt in Vs:
        Vp = Vt[piv]
        prow, pcols = _pivots(Vp, p)
        if len(prow) != rank:
            raise RuntimeError(f"pivot rows dependent modulo {p}: projection lost rank")
        Ct = _matmul_mod(Vt[:, pcols], B.inverse_mod(Vp[:, pcols], p), p)
        w = rng.integers(0, p, size=(Vt.shape[1], 2))
        if not np.array_equal(_matmul_mod(Ct, _matmul_mod(Vp, w, p), p), _matmul_mod(Vt, w, p)):
            raise RuntimeError(f"a form is outside the pivot span modulo {p}")
        if (Ct[later] != 0).any():
            raise RuntimeError(f"pivot structure differs modulo {p}")
        C.append(Ct[np.ix_(need, cols)])
    return (C[0] + C[1]) % p * pow(2, -1, p) % p, (C[0] - C[1]) % p * pow(2 * s, -1, p) % p


def _reconstruct(res: list, qs: list):
    """Exact Q(sqrt2) coefficients from all but the last prime, checked on the last; None on failure."""
    n_rec = len(qs) - 1
    m = prod(qs[:n_rec])
    wts = [(m // q) * pow(m // q, -1, q) for q in qs[:n_rec]]
    exact = {}
    nzmask = np.any([x != 0 for r in res[:n_rec] for x in r], axis=0)
    if np.any([(x != 0) & ~nzmask for x in res[-1]]):  # zero for every other prime: zero on the held-out one
        return None
    nz = np.argwhere(nzmask)
    for i, c in nz:
        pair = []
        for part in (0, 1):
            x = sum(int(res[t][part][i, c]) * wts[t] for t in range(n_rec)) % m
            f = B.ratrec(x, m)
            if f is None or (f.numerator - f.denominator * int(res[-1][part][i, c])) % qs[-1]:
                return None
            pair.append(f)
        exact[(int(i), int(c))] = tuple(pair)
    return exact


def compile_hierarchy(h: Mo.Hierarchy, workers: int = 1, K: int = 2048, seed: int = 20261007, log=print,
                      cache: Path | None = None, only: list | None = None) -> Problem | None:
    """The exact problem of a hierarchy.  The rank is found modulo the first prime with K functionals
    (doubled until it exceeds rank + 64); the other primes use rank + 64 of them, and are added until the
    reconstruction verifies on a held-out one.  The projections of every prime are kept in `cache` (a
    directory), so that an interrupted compile resumes without projecting again.  With `only` (prime
    indices), just those primes are projected into the cache and None is returned: independent jobs can
    share the primes of one compile, which then runs from the cache."""
    t0 = time.time()
    ps = Pa.primes(MAX_PRIMES)
    blocks = list(h.rows.items())
    obs_names = list(Mo.OBSERVABLES) + ["P2", "P4"]
    order = [("ward", w[2]) for w in h.ward] + [("one",)] + [("obs", o) for o in obs_names] + \
            [("entry", lam, i, j) for lam, rs in blocks for i in range(len(rs)) for j in range(i, len(rs))]
    n_ward = len(h.ward)
    labels = [_label(k) for k in order]

    def images(sel, K_):
        """Per prime of sel, the (F, 4, K_) projections of the forms in order (cached)."""
        files = {t: cache / f"projection-{t}-{K_}.npy" if cache else None for t in sel}   # checked below
        todo = [t for t in sel if not (files[t] and files[t].exists())]
        if todo:
            forms = project(h, [ps[t] for t in todo], K_, seed, workers, log)
            forms[("obs", "P2")] = forms[("obs", "T")] * 2 % np.array([ps[t][0] for t in todo], dtype=np.int32)[:, None, None]
            one = np.zeros((len(todo), 4, K_), dtype=np.int32)
            one[:, 0] = 1
            forms[("one",)] = one
            for n, t in enumerate(todo):
                arr = np.stack([forms[k][n] for k in order])
                if cache:
                    cache.mkdir(parents=True, exist_ok=True)
                    np.save(files[t], arr)
                files[t] = files[t] or arr
            del forms
        arrs = [np.load(files[t]) if isinstance(files[t], Path) else files[t] for t in sel]
        if any(a.shape != (len(order), 4, K_) for a in arrs):    # (and every prime is checked by the reduction)
            raise RuntimeError("cached projections do not match this hierarchy")
        return _images(arrs, [ps[t] for t in sel], K_)

    while True:
        V0 = images([0], K)[0]
        piv, _ = _pivots(V0[0], ps[0][0])
        rank, id_row = len(piv), 2 * n_ward
        if rank <= K - 64:
            break
        log(json.dumps({"event": "rank", "rank": rank, "K": K, "retry": 2 * K}))
        K *= 2
    if id_row not in piv:
        raise RuntimeError("the identity lies in the span of the Ward forms: the hierarchy has no solution")
    K1 = rank + 64
    V0 = [V[:, :K1] for V in V0]
    if _pivots(V0[0], ps[0][0])[0] != piv:
        raise RuntimeError("the first rank + 64 functionals do not separate the forms")
    if only is not None:
        images(only, K1)
        log(json.dumps({"event": "projected", "primes": only, "K": K1, "seconds": round(time.time() - t0, 1)}))
        return None
    ward_piv = [c for c, r in enumerate(piv) if r < id_row]
    free = [c for c, r in enumerate(piv) if r > id_row]
    id_col = piv.index(id_row)
    log(json.dumps({"event": "pivots", "rank": rank, "ward_rank": len(ward_piv), "dimension": len(free),
                    "rows": len(V0[0]), "K": K1, "seconds": round(time.time() - t0, 1)}))
    need, cols = np.arange(id_row, len(V0[0])), [id_col] + free
    rng = np.random.default_rng(seed + 1)
    res = [_coefficients(V0, ps[0], piv, need, cols, rng)]
    del V0
    exact, batch = None, 6
    while exact is None:
        if len(res) + batch > MAX_PRIMES:
            raise RuntimeError(f"rational reconstruction failed with {len(res)} primes")
        sel = list(range(len(res), len(res) + batch))
        for t, Vs in zip(sel, images(sel, K1)):
            res.append(_coefficients(Vs, ps[t], piv, need, cols, rng))
        exact = _reconstruct(res, [ps[t][0] for t in range(len(res))])
        log(json.dumps({"event": "reconstruction", "primes": len(res), "ok": exact is not None,
                        "seconds": round(time.time() - t0, 1)}))
        batch = 4
    qs = [ps[t][0] for t in range(len(res))]
    n_rec = len(qs) - 1
    log(json.dumps({"event": "reconstructed", "nonzero": len(exact), "bits": prod(qs[:n_rec]).bit_length(),
                    "seconds": round(time.time() - t0, 1)}))

    zero = (Fraction(0), Fraction(0))
    by_row = {}
    for (i, c), v in exact.items():
        by_row.setdefault(i, {})[c] = v

    def affine(row):
        terms = by_row.get(row - id_row, {})
        return terms.get(0, zero), [(z, terms[1 + z]) for z in range(len(free)) if 1 + z in terms]

    observables = {name: affine(2 * (n_ward + 1 + i)) for i, name in enumerate(obs_names)}
    out, f = [], n_ward + 1 + len(obs_names)
    for lam, rs in blocks:
        ents = []
        for i in range(len(rs)):
            for j in range(i, len(rs)):
                ents.append((i, j, affine(2 * f), affine(2 * f + 1)))
                f += 1
        out.append(Block(Mo.family(lam), [r.name for r in rs], ents))
    out, removed = facial(out, ps[:2])
    log(json.dumps({"event": "facial", "removed": removed, "seconds": round(time.time() - t0, 1)}))
    coords = [f"{labels[piv[c] // 2]} ({'Re' if piv[c] % 2 == 0 else 'Im'})" for c in free]
    manifest = h.manifest()
    manifest["removed_rows"] = removed
    prov = {"primes": qs, "reconstruction_primes": n_rec, "K": K1, "seed": seed, "rank": rank,
            "ward_rank": len(ward_piv), "forms": len(order), "code_sha256": code_hashes(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    return Problem(h.level, manifest, coords, out, observables, prov)


def _label(k) -> str:
    if k[0] == "ward":
        return k[1]
    if k[0] == "one":
        return "<1>"
    if k[0] == "obs":
        return f"<{k[1]}>"
    return f"{Mo.family(k[1])}: <{k[2]}|{k[3]}>"


def facial(blocks: list, ps) -> tuple:
    """Remove from each block the common kernel of its pencil M_0 + sum_z z M_z (found modulo two primes,
    which must agree): the rows of the free columns of the stacked pencil's echelon form."""
    out, removed = [], {}
    rng = np.random.default_rng(5)
    for b in blocks:
        n = len(b.rows)
        frees = []
        for p, ip, s in ps:
            mats = {}
            for i, j, re, im in b.entries:
                for key, cre, cim in [("c", re[0], im[0])] + [(z, v, None) for z, v in re[1]] + [(z, None, v) for z, v in im[1]]:
                    Mx = mats.setdefault(key, np.zeros((n, n), dtype=np.int64))
                    if cre is not None and cre != (0, 0):
                        x = B.zmod(cre, p, s)
                        Mx[i, j] = (Mx[i, j] + x) % p
                        if i != j:
                            Mx[j, i] = (Mx[j, i] + x) % p
                    if cim is not None and cim != (0, 0):
                        y = B.zmod(cim, p, s) * ip % p
                        Mx[i, j] = (Mx[i, j] + y) % p
                        if i != j:
                            Mx[j, i] = (Mx[j, i] - y) % p
            stack = np.vstack(list(mats.values())) if mats else np.zeros((0, n), dtype=np.int64)
            if len(stack) > n + 8:                     # residues < 2^20: exact int64 sums
                stack = rng.integers(-1, 2, size=(n + 8, len(stack))) @ stack % p
            _, piv = B.rref_mod(stack, p)
            frees.append([c for c in range(n) if c not in piv])
        if frees[0] != frees[1]:
            raise RuntimeError(f"{b.family}: common kernel differs between primes")
        drop = set(frees[0])
        if drop:
            removed[b.family] = [b.rows[c] for c in sorted(drop)]
        keep = [c for c in range(n) if c not in drop]
        new = {c: k for k, c in enumerate(keep)}
        if keep:
            out.append(Block(b.family, [b.rows[c] for c in keep],
                             [(new[i], new[j], re, im) for i, j, re, im in b.entries if i in new and j in new]))
    return out, removed
