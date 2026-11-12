"""Solvers for the exact problem.

  minimize  c(z) = sum_o w_o <o>(z)   subject to   M_b(z) >= 0 for every block b,

optionally with exact constraints <o>(z) = v, eliminated by substituting one
coordinate in Q(sqrt2).  Complex blocks are realified as [[Re, -Im], [Im, Re]].
SDPB gives bounds (from the residues of its certificate) and points checked at high precision
against the exact data; Clarabel (double precision) is for exploration and its results are
labeled unverified.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np
from scipy import sparse

from .problem import Problem

ZERO = (Fraction(0), Fraction(0))
SDPB_BIN = os.environ.get("SDPB_BIN", "")      # directory of sdpb and pmp2sdp; empty: found on PATH
SDPB_IMAGE = os.environ.get("SDPB_IMAGE", "")  # or a Singularity image of SDPB (its own mpirun, one machine)


# ---- Q(sqrt2) numbers: pairs (a, b) = a + b sqrt2
def qadd(u, v):
    """Sum of two numbers a + b sqrt2."""
    return (u[0] + v[0], u[1] + v[1])


def qmul(u, v):
    """Product of two numbers a + b sqrt2."""
    return (u[0] * v[0] + 2 * u[1] * v[1], u[0] * v[1] + u[1] * v[0])


def qinv(u):
    """Inverse of a nonzero number a + b sqrt2."""
    n = u[0] * u[0] - 2 * u[1] * u[1]
    return (u[0] / n, -u[1] / n)


def qfloat(u) -> float:
    """A number a + b sqrt2 as a float."""
    return float(u[0]) + float(u[1]) * 2 ** 0.5


# ---- the real SDP: affine forms are (constant, {coordinate: coefficient})
@dataclass
class SDP:
    blocks: list            # [(size, {(i, j): affine} for i <= j)]
    objective: tuple        # affine
    coordinates: list[str]
    observables: dict       # {name: affine}, to locate solutions in observable planes
    note: str = ""


def _aff(a):
    """Mutable copy of an affine form."""
    return (a[0], dict(a[1]))


def _neg(a):
    """Negative of an affine form."""
    return (-a[0][0], -a[0][1]), {q: (-v[0], -v[1]) for q, v in a[1].items()}


def _substitute(a, k, expr):
    """Replace z_k by the affine expr in the affine a."""
    c = a[1].get(k)
    if c is None:
        return a
    const, lin = qadd(a[0], qmul(c, expr[0])), {q: v for q, v in a[1].items() if q != k}
    for q, v in expr[1].items():
        w = qadd(lin.get(q, ZERO), qmul(c, v))
        if w == ZERO:
            lin.pop(q, None)
        else:
            lin[q] = w
    return const, lin


def specialize(prob: Problem, objective: dict, fixed: dict | None = None) -> SDP:
    """Exact real SDP for one objective {observable: weight} and constraints {observable: value}."""
    obs = {k: _aff(v) for k, v in prob.observables.items()}
    blocks = [[(i, j, _aff(re), _aff(im)) for i, j, re, im in b.entries] for b in prob.blocks]
    for name, value in (fixed or {}).items():
        const, lin = obs[name]
        if not lin:
            raise ValueError(f"<{name}> is a constant of the problem")
        k = min(lin, key=lambda q: (lin[q][1] != 0, q))               # prefer a rational coefficient
        inv = qinv(lin[k])
        rhs = qadd((Fraction(value), Fraction(0)), (-const[0], -const[1]))
        expr = (qmul(rhs, inv), {q: qmul((-v[0], -v[1]), inv) for q, v in lin.items() if q != k})
        obs = {n: _substitute(a, k, expr) for n, a in obs.items()}
        blocks = [[(i, j, _substitute(re, k, expr), _substitute(im, k, expr)) for i, j, re, im in b] for b in blocks]
    used = sorted({q for b in blocks for _, _, re, im in b for q in list(re[1]) + list(im[1])})
    free = {n for n, a in obs.items() if not set(a[1]) <= set(used)}     # not determined by the blocks
    if free & set(objective):
        raise ValueError(f"{sorted(free & set(objective))} not constrained by the blocks at this level")
    obs = {n: a for n, a in obs.items() if n not in free}
    new = {q: n for n, q in enumerate(used)}

    def ren(a):
        return a[0], {new[q]: v for q, v in a[1].items()}

    out = []
    for b, ents in zip(prob.blocks, blocks):
        n = len(b.rows)
        cplx = any(im[1] or im[0] != ZERO for *_, im in ents)
        M = {}
        for i, j, re, im in ents:
            M[(i, j)] = ren(re)
            if cplx:
                M[(n + i, n + j)] = ren(re)
                if i != j:
                    M[(i, n + j)] = ren(_neg(im))                # [[Re, -Im], [Im, Re]]
                    M[(j, n + i)] = ren(im)
        out.append((2 * n if cplx else n, M))
    obj = (ZERO, {})
    for name, w in objective.items():
        a = ren(obs[name])
        wq = (Fraction(w), Fraction(0))
        obj = (qadd(obj[0], qmul(wq, a[0])), obj[1])
        for q, v in a[1].items():
            obj[1][q] = qadd(obj[1].get(q, ZERO), qmul(wq, v))
    return SDP(out, obj, [prob.coordinates[q] for q in used], {n: ren(a) for n, a in obs.items()},
               note=json.dumps({"objective": {k: str(w) for k, w in objective.items()}, "fixed": {k: str(v) for k, v in (fixed or {}).items()}}))


def _cap_block(sdp: SDP, cap: int) -> tuple:
    """The 1x1 block cap - sum_b tr M_b(z) >= 0."""
    const, lin = (Fraction(cap), Fraction(0)), {}
    for n, M in sdp.blocks:
        for i in range(n):
            if (i, i) in M:
                c, l = _neg(M[(i, i)])
                const = qadd(const, c)
                for q, v in l.items():
                    lin[q] = qadd(lin.get(q, ZERO), v)
    return 1, {(0, 0): (const, lin)}


def capped(sdp: SDP, cap: int) -> SDP:
    """sdp with sum_b tr M_b(z) <= cap.  The feasible set becomes compact, so the optimum is attained
    also where the infimum of sdp needs diverging moments.  Its solutions are feasible points of sdp;
    its optimal value is not a bound for sdp."""
    return SDP(sdp.blocks + [_cap_block(sdp, cap)], sdp.objective, sdp.coordinates, sdp.observables,
               note=json.dumps({**json.loads(sdp.note or "{}"), "cap": str(cap)}))


def strictness(sdp: SDP, trace_cap: int = 10 ** 12) -> SDP:
    """Diagnostic SDP: maximize t with M_b(z) - t 1 >= 0 and sum_b tr M_b(z) <= trace_cap."""
    t = len(sdp.coordinates)
    blocks = []
    for n, M in sdp.blocks:
        M = {ij: (a[0], dict(a[1])) for ij, a in M.items()}
        for i in range(n):
            M.setdefault((i, i), (ZERO, {}))[1][t] = (Fraction(-1), Fraction(0))
        blocks.append((n, M))
    return SDP(blocks + [_cap_block(sdp, trace_cap)], (ZERO, {t: (Fraction(-1), Fraction(0))}),
               sdp.coordinates + ["t"], sdp.observables, note="strict feasibility")


def evaluate(a, z) -> float:
    """Value of an affine form at z (double precision)."""
    return qfloat(a[0]) + sum(qfloat(v) * float(z[q]) for q, v in a[1].items())


# ---- numeric evaluation
def matrices(sdp: SDP, num=qfloat):
    """Per block: list [F_0, F_1, ..., F_m] of full symmetric matrices (entries num(q))."""
    m = len(sdp.coordinates)
    out = []
    for n, M in sdp.blocks:
        zero = num(ZERO)
        F = [[[zero] * n for _ in range(n)] for _ in range(m + 1)]
        for (i, j), (c, lin) in M.items():
            for k, v in [(0, c)] + [(q + 1, w) for q, w in lin.items()]:
                F[k][i][j] = F[k][j][i] = num(v)
        out.append(F)
    return out


def min_eigs(sdp: SDP, z) -> list[float]:
    """Smallest eigenvalue of every block at z (double precision)."""
    out = []
    for F in matrices(sdp):
        A = np.array(F[0], dtype=float) + sum(float(zk) * np.array(Fk, dtype=float) for zk, Fk in zip(z, F[1:]))
        out.append(float(np.linalg.eigvalsh(A)[0]))
    return out


# ---- Clarabel (exploration)
def solve_clarabel(sdp: SDP, tol: float = 1e-9, max_iter: int = 500, verbose: bool = False) -> dict:
    import clarabel
    m = len(sdp.coordinates)
    rows, cols, vals, b, cones, r0 = [], [], [], [], [], 0
    r2 = 2 ** 0.5
    for n, M in sdp.blocks:
        for j in range(n):
            for i in range(j + 1):                      # svec: upper triangle, column major, sqrt2 off-diagonal
                s = 1.0 if i == j else r2
                c, lin = M.get((i, j), (ZERO, {}))
                b.append(s * qfloat(c))
                for q, v in lin.items():
                    rows.append(r0)
                    cols.append(q)
                    vals.append(-s * qfloat(v))
                r0 += 1
        cones.append(clarabel.PSDTriangleConeT(n))
    A = sparse.csc_matrix((vals, (rows, cols)), shape=(r0, m))
    q = np.zeros(m)
    for k, v in sdp.objective[1].items():
        q[k] = qfloat(v)
    st = clarabel.DefaultSettings()
    st.verbose, st.tol_gap_abs, st.tol_gap_rel, st.tol_feas, st.max_iter = verbose, tol, tol, tol, max_iter
    sol = clarabel.DefaultSolver(sparse.csc_matrix((m, m)), q, A, np.array(b), cones, st).solve()
    z = list(sol.x)
    c0 = qfloat(sdp.objective[0])
    return {"backend": "clarabel", "status": str(sol.status), "value": c0 + float(q @ np.array(z)),
            "primal": c0 + sol.obj_val, "dual": c0 + sol.obj_val_dual, "z": [repr(v) for v in z],
            "min_eigs": min_eigs(sdp, z), "verified": False}


# ---- SDPB (bounds)
def write_pmp(sdp: SDP, path: Path, digits: int = 300) -> None:
    """SDPB polynomial-matrix program over y = (1, z): maximize -c.y with normalization y_0 = 1."""
    mpmath.mp.dps = digits + 20
    r2 = mpmath.sqrt(2)

    def num(u):
        if u == ZERO:
            return "0"
        x = mpmath.mpf(u[0].numerator) / u[0].denominator + mpmath.mpf(u[1].numerator) / u[1].denominator * r2
        return mpmath.nstr(x, digits, min_fixed=1, max_fixed=0)

    m = len(sdp.coordinates)
    obj = [sdp.objective[0]] + [sdp.objective[1].get(k, ZERO) for k in range(m)]
    pmp = {"objective": [num((-u[0], -u[1])) for u in obj],
           "normalization": ["1"] + ["0"] * m,
           "PositiveMatrixWithPrefactorArray": [
               {"polynomials": [[[[v] for v in (F[k][i][j] for k in range(m + 1))] for j in range(len(F[0]))]
                                for i in range(len(F[0]))]}
               for F in matrices(sdp, num)]}
    path.write_text(json.dumps(pmp, separators=(",", ":")))


def _last_iterate(out: Path) -> dict:
    """Termination reason (out.txt, absent if SDPB aborted) and the last iterate (iterations.json)."""
    txt = (out / "out.txt").read_text() if (out / "out.txt").exists() else ""
    m = re.search(r'terminateReason = "([^"]*)"', txt)
    raw = (out / "iterations.json").read_text().rstrip().rstrip(",]")
    last = json.loads(raw[raw.rindex("{"):])
    return {"terminate": m.group(1) if m else "aborted", "iterations": last["iteration"],
            **{k: last[k] for k in ("P-obj", "D-obj", "P-err", "p-err", "D-err")}}


def verify(sdp: SDP, z: list[str], dps: int) -> list[str]:
    """Smallest eigenvalue of every block at z, from the exact data at dps digits."""
    mpmath.mp.dps = dps
    r2 = mpmath.sqrt(2)
    zs = [mpmath.mpf(v) for v in z]

    def num(u):
        return mpmath.mpf(u[0].numerator) / u[0].denominator + mpmath.mpf(u[1].numerator) / u[1].denominator * r2

    out = []
    for n, M in sdp.blocks:
        A = mpmath.zeros(n, n)
        for (i, j), (c, lin) in M.items():
            A[i, j] = A[j, i] = num(c) + mpmath.fsum(num(v) * zs[q] for q, v in lin.items())
        out.append(mpmath.nstr(min(mpmath.eigsy(A, eigvals_only=True)), 8))
    return out


def solve_sdpb(sdp: SDP, work: Path, precision: int = 1024, procs: int = 1, threshold: float = 1e-30,
               max_iter: int = 1000) -> dict:
    """Solve with SDPB at the given precision (bits).

    By weak duality, value = -P-obj is a rigorous lower bound ("bound") as soon as SDPB's primal
    residuals (of the certificate) are <= thr, whatever the duality gap, also when SDPB stops early
    because the infimum is not attained (moments diverging along the valley).  The gap |P - D|
    is recorded: it says how close the bound is to the optimum.  The returned point is "feasible"
    if every block at it has smallest eigenvalue >= -1e3 thr from the exact data, however SDPB
    stopped.  The result is "verified" (an optimal feasible point) if moreover SDPB reports
    optimality and the gap is <= 1e3 thr (1 + |P|)."""
    t = [time.time()]
    for old in ("out", "ck", "sdp"):                 # a fresh start: never resume another run's checkpoint
        shutil.rmtree(work / old, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    write_pmp(sdp, work / "pmp.json", digits=int(precision * 0.29))
    t.append(time.time())
    def run(np_, *a):
        threads = os.environ.get("SDPB_THREADS", "1")
        cmd = ["mpirun", "--oversubscribe", "--bind-to", "none", "-np", str(np_), *a]
        if SDPB_IMAGE:              # no host environment: the container's own MPI launches the processes
            cmd = ["singularity", "exec", "--cleanenv", "--env", f"OMP_NUM_THREADS={threads}", SDPB_IMAGE, *cmd]
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": threads})
        if r.returncode:
            raise RuntimeError(f"{a[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    run(1, os.path.join(SDPB_BIN, "pmp2sdp"), "--input", str(work / "pmp.json"), "--output", str(work / "sdp"),
        "--precision", str(precision))
    try:
        run(procs, os.path.join(SDPB_BIN, "sdpb"), "--sdpDir", str(work / "sdp"), "--outDir", str(work / "out"),
            "--checkpointDir", str(work / "ck"), "--precision", str(precision), "--maxIterations", str(max_iter),
            "--dualityGapThreshold", str(threshold), "--primalErrorThreshold", str(threshold),
            "--dualErrorThreshold", str(threshold), "--writeSolution", "x,y,X", "--noFinalCheckpoint")
    except RuntimeError as err:
        if not (work / "out" / "iterations.json").exists():
            raise
        abort = str(err)[-400:]
    else:
        abort = None
    t.append(time.time())
    mpmath.mp.dps = precision * 3 // 10
    it = _last_iterate(work / "out")
    p, d = mpmath.mpf(it["P-obj"]), mpmath.mpf(it["D-obj"])
    thr = mpmath.mpf(threshold)
    # the certificate (SDPB's primal) alone makes -P-obj a rigorous lower bound
    bound = mpmath.mpf(it["P-err"]) <= thr and mpmath.mpf(it["p-err"]) <= thr
    z, eigs = [], []
    if (work / "out" / "y.txt").exists() and it["terminate"] != "aborted":
        y = (work / "out" / "y.txt").read_text().split()
        z = y[2:2 + int(y[0])]
        eigs = verify(sdp, z, precision * 3 // 10)
    t.append(time.time())
    feasible = bool(eigs) and all(mpmath.mpf(e) >= -1000 * thr for e in eigs)
    ok = (it["terminate"] == "found primal-dual optimal solution" and bound and feasible
          and abs(p - d) <= 1000 * thr * (1 + abs(p)))
    return {"backend": "sdpb", "status": it["terminate"], "value": mpmath.nstr(-p, 40),
            "primal": mpmath.nstr(-p, 40), "dual": mpmath.nstr(-d, 40), "errors": [mpmath.nstr(mpmath.mpf(it[k]), 3) for k in ("P-err", "p-err", "D-err")],
            "iterations": it["iterations"], "z": z, "min_eigs": eigs, "feasible": feasible, "verified": bool(ok), "bound": bool(bound),
            "gap": mpmath.nstr(abs(p - d), 3), "precision": precision, "threshold": threshold,
            "abort": abort, "note": sdp.note,
            "seconds": {"write": round(t[1] - t[0], 1), "sdpb": round(t[2] - t[1], 1), "verify": round(t[3] - t[2], 1)}}
