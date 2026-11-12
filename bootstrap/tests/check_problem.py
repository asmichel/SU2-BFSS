"""Exact checks of the forms of a level, block by block, against explicit products.

For sample pairs in every block (Gram pairs of rows and Ward pairs of a row with a null multiplet) the
pairing pair(h, h') = sum_k o_k(h)^+ * o_k(h') / D_k is computed as an explicit polynomial (exact star
products of the components), and checked to be invariant (annihilated by every simple raising and lowering
derivation) and Hermitian (pair(h', h) = pair(h, h')^+); its projections are compared with those of the
fast evaluation (pairing.Pairing) modulo two primes.

  python tests/check_problem.py LEVEL [MAX_DIM] [WORKERS]
"""
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bfss import basis as B, moments as Mo, pairing as Pa


def explicit(irr, ca, cb) -> dict:
    """pair(h, h') as an explicit polynomial, from exact star products of the components."""
    out = {}
    for (D, _), oa, ob in zip(irr.comps, ca, cb):
        for m, z in Pa.star(Pa.dagger(oa), ob).items():
            out[m] = B.zadd(out.get(m, B.Z0), (z[0] / D, z[1] / D))
    return {m: z for m, z in out.items() if z != B.Z0}


def project(P: dict, fun) -> np.ndarray:
    """The functionals of fun applied to an explicit polynomial."""
    ref = np.zeros(fun.K, dtype=np.int64)
    for (xs, ps, fs), z in P.items():
        c = fun.z(z)
        nd = 1
        for v in Pa._rest(xs, ps, fs):
            nd = nd * int(fun.vals[v]) % fun.p
        for j in range(fun.K):
            ref[j] = (ref[j] + c * pow(nd, j, fun.p)) % fun.p
    return ref


def check_block(lam):
    t = time.time()
    rng = random.Random(hash(lam) % 1000)
    rows = sorted(h.rows[lam], key=lambda r: len(r.poly))
    copies = [r.poly for r in rows[:3]]
    nulls = [n for l2, parts in h.nulls for n in parts if l2 == lam and n]
    if nulls:
        copies.append(min(nulls, key=len))
    irr = Pa.Irrep(lam, copies[0])
    comps = [irr.components(c) for c in copies]
    pairs = [(a, b) for a in range(len(copies)) for b in range(len(copies))]
    pairs = rng.sample(pairs, min(5, len(pairs)))
    act = B.action()
    bad = []
    for prime in Pa.primes(2):
        fun = Pa.Functionals(prime, 6, 3)
        G = Pa.Pairing([[Pa.dagger(o) for o in c] for c in comps], comps, irr).evaluate(fun)
        for a, b in pairs:
            P = explicit(irr, comps[a], comps[b])
            if (project(P, fun) != G[a, b]).any():
                bad.append(f"projection {a},{b}")
    for a, b in pairs[:2]:
        P, Q = explicit(irr, comps[a], comps[b]), explicit(irr, comps[b], comps[a])
        if any(B.derive(P, E) for E in act["raise"] + act["lower"]):
            bad.append(f"not invariant {a},{b}")
        if Pa.dagger(P) != Q:
            bad.append(f"not Hermitian {a},{b}")
    return Mo.family(lam), B.dimension(lam), len(copies), bool(nulls), bad, round(time.time() - t, 1)


if __name__ == "__main__":
    LEVEL = int(sys.argv[1])
    MAX_DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    h = Mo.canonical(LEVEL, WORKERS, log=lambda s: None)
    lams = [l for l in h.rows if B.dimension(l) <= MAX_DIM]
    results = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for fam, dim, n, has_null, bad, sec in pool.imap_unordered(check_block, lams):
            results.append(bad)
            print(json.dumps({"family": fam, "dim": dim, "copies": n, "null": has_null, "bad": bad, "seconds": sec}), flush=True)
    print(f"level {LEVEL}: {len(results)} blocks checked, {sum(1 for r in results if r)} with failures")
