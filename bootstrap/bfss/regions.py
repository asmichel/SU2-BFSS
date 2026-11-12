"""Allowed regions in observable planes.

For a plane (X, Y) with window [x0, x1] x [y0, y1] the support solve at angle phi
minimizes cos(phi) X/dx + sin(phi) Y/dy (dx, dy the window spans).  Its certificate
gives a half-plane containing the allowed region (outer set), and its optimum, when
attained, a feasible point.  Edge solves fix X or Y at a window edge and minimize or
maximize the other coordinate; they add feasible points where the region leaves the
window.  Where an infimum is not attained (moments diverging along the valley), the
same solve with a cap on the total trace gives feasible points that approach it.
Bounds of lower levels hold at higher ones (the hierarchy is nested), so they join the
outer set.  The inner set is the convex hull of the feasible points.  The region is
convex, so both sets converge; every vertex of the outer polygon (clipped to the
window) must lie within TOLERANCE (in window units) of the inner set.  Plots read only
the file written here.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from . import sdp as S
from .problem import Problem

HALF = np.pi / 2
PLANES = {   # X, Y, x window, y window, support directions (phi_lo, phi_hi) with a finite minimum
    "R2_u": ("R2", "u", (7.2, 12.0), (8.0, 25.5), (-HALF, HALF)),          # open towards +X
    "R2_R4": ("R2", "R4", (7.2, 12.0), (50.0, 155.0), (0.0, np.pi)),      # open towards +Y
    "R4_u": ("R4", "u", (50.0, 120.0), (8.0, 25.5), (-HALF, HALF)),
}
TOLERANCE = 1e-3
# A certified bound holds up to |residual| |z| at a point z: exact for moderate moments, but feasible points
# on non-attained edges (|z| ~ 1e50) may graze it by ~1e-6.  Larger violations flag an inconsistency.
CONSISTENCY = 1e-5
CAPS = (10 ** 18, 10 ** 21)     # trace caps, tried in order, for infima that are not attained


def directions(n: int, plane: str) -> list:
    """n support directions spanning the plane's closed range of bounded directions.  Outside it the
    minimum is -infinity; SDPB can still return a certificate with tiny residues there (the problem is
    weakly infeasible), whose value is no bound."""
    return [float(phi) for phi in np.linspace(*PLANES[plane][4], n)]


def in_range(phi: float, plane: str) -> bool:
    lo, hi = PLANES[plane][4]
    return lo - 1e-9 <= phi <= hi + 1e-9


def edges(plane: str) -> list:
    """(fixed axis, its window value, sense of the other axis): both extremes on all four edges."""
    X, Y, (x0, x1), (y0, y1), _ = PLANES[plane]
    return [("edge", ax, v, sense) for ax, vals in ((0, (x0, x1)), (1, (y0, y1))) for v in vals for sense in (1, -1)]


# A solve is specified by a direction phi, an edge ("edge", axis, value, sense), or ("cap", spec, cap).
def _spec(x):
    """The hashable spec of a JSON-decoded one."""
    if isinstance(x, (list, tuple)) and x[0] == "cap":
        return "cap", _spec(x[1]), int(x[2])
    return tuple(x) if isinstance(x, (list, tuple)) else float(x)


def spec_of(res: dict):
    """The spec a cached result was solved for."""
    if res.get("corner"):
        return ("corner", float(res["corner"][0]), float(res["corner"][1]))
    base = res["phi"] if res.get("phi") is not None else _spec(res["edge"])
    return ("cap", base, int(res["cap"])) if res.get("cap") else base


def _key(plane: str, spec) -> str:
    """File name of a solve: the plane plus the direction, the edge, and the cap."""
    if isinstance(spec, tuple) and spec[0] == "cap":
        return f"{_key(plane, spec[1])}_cap{spec[2]:.0e}"
    if isinstance(spec, float):
        return f"{plane}_{spec:+.9f}"
    if spec[0] == "corner":
        return f"{plane}_corner_{spec[1]:g}_{spec[2]:g}"
    _, ax, v, sense = spec
    return f"{plane}_edge_{'XY'[ax]}{v:g}_{'min' if sense > 0 else 'max'}"


_CTX: dict = {}


def _solve(task):
    """Worker: one support, edge or capped solve, cached on disk."""
    plane, spec = task
    cache = _CTX["run"] / "solves" / "scan" / f"{_key(plane, spec)}.{_CTX['backend']}.json"
    try:                                           # reuse only definitive results
        res = json.loads(cache.read_text())
        st = str(res.get("status"))
        if (res.get("precision") == _CTX["precision"] or res.get("verified") or is_bound(res)
                or "Infeasible" in st or "maxComplementarity" in st):
            return res                             # re-solving at the same precision gives the same answer
    except (OSError, ValueError):                  # missing, or cut short by a killed job
        pass
    X, Y, (x0, x1), (y0, y1), _ = PLANES[plane]
    corner = spec[1:] if isinstance(spec, tuple) and spec[0] == "corner" else None
    cap = CAPS[0] if corner else spec[2] if isinstance(spec, tuple) and spec[0] == "cap" else None
    base = None if corner else spec[1] if cap else spec
    if corner:                                     # any feasible point at a window corner (zero objective)
        obj, fixed = {}, {X: Fraction(corner[0]), Y: Fraction(corner[1])}
    elif isinstance(base, float):
        obj, fixed = {X: Fraction(np.cos(base) / (x1 - x0)), Y: Fraction(np.sin(base) / (y1 - y0))}, None
    else:
        _, ax, v, sense = base
        obj, fixed = {(Y, X)[ax]: Fraction(sense)}, {(X, Y)[ax]: Fraction(v)}
    try:
        sd = S.specialize(_CTX["problem"], obj, fixed)
        if cap:                                    # bounded by construction: straight to SDPB
            sd = S.capped(sd, cap)
            res = {"status": "not solved"}
        else:
            res = S.solve_clarabel(sd)             # cheap screen: unbounded directions never reach SDPB
        if _CTX["backend"] == "sdpb" and "Infeasible" not in res["status"]:
            res = S.solve_sdpb(sd, _CTX["run"] / "sdpb" / _key(plane, spec), precision=_CTX["precision"],
                               threshold=_CTX["threshold"], procs=_CTX["procs"], max_iter=_CTX["max_iter"])
        if res.get("z"):
            res["point"] = {k: S.evaluate(a, res["z"]) for k, a in sd.observables.items()}
    except Exception as err:                       # solver failures are recorded, never plotted
        res = {"status": f"error: {err}"[:500], "verified": False}
    res.update(plane=plane, phi=base if isinstance(base, float) else None,
               edge=base if isinstance(base, tuple) else None, cap=cap, corner=corner)
    res.pop("z", None)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")                # atomic: a killed job never leaves half a file
    tmp.write_text(json.dumps(_finite(res), default=str, allow_nan=False))
    tmp.replace(cache)
    return res


def load_solves(run: Path) -> list:
    """Every cached solve of a run (support, edge and capped solves of all planes)."""
    out = []
    for f in sorted((run / "solves" / "scan").glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    old = sorted({r.get("plane") for r in out} & {"R2_V", "R4_V"})
    if old:                                        # records from before the observables were named u and Y
        raise ValueError(f"{run}: solves in the planes {old}; rename R2_V, R4_V, V and F to R2_u, R4_u, u and Y")
    return out


def scan(prob: Problem, run: Path, planes, n: int, backend: str, workers: int, precision: int = 1024,
         threshold: float = 1e-30, rounds: int = 8, lower: list = (), log=print, procs: int = 1,
         max_iter: int = 1000) -> list:
    """n directions and the edge solves per plane, then the solves that every outer vertex not yet
    within TOLERANCE of the region asks for, until none is left (or `rounds` rounds).  A resumed scan
    starts from all solves cached in the run.  `lower` holds the solves of lower levels of the same
    nested hierarchy; their bounds hold here too."""
    _CTX.update(problem=prob, backend=backend, run=run, precision=precision, threshold=threshold, procs=procs,
                max_iter=max_iter)
    tasks = {_key(pl, s): (pl, s) for pl in planes for s in directions(n, pl) + edges(pl)}
    for r in load_solves(run):
        if r["plane"] in planes and (r.get("phi") is None or in_range(r["phi"], r["plane"])):
            tasks.setdefault(_key(r["plane"], spec_of(r)), (r["plane"], spec_of(r)))
    tasks, results, done = list(tasks.values()), [], set()
    with mp.get_context("fork").Pool(workers) as pool:
        for r in range(rounds + 1):
            results += pool.map(_solve, tasks)
            done |= {_key(pl, s) for pl, s in tasks}
            geo = {pl: geometry(results, pl, lower) for pl in planes}
            tasks = list({_key(pl, _spec(s)): (pl, _spec(s)) for pl in planes for s in geo[pl].get("next", [])
                          if _key(pl, _spec(s)) not in done}.values())
            log(json.dumps({"round": r, "solves": len(results), "gaps": {pl: g.get("gap") for pl, g in geo.items()},
                            "next": len(tasks)}), flush=True)
            if not tasks:
                break
    return results


def is_bound(res: dict) -> bool:
    """A rigorous support value (half-plane): SDPB's certificate converged.  Capped solves give points only."""
    if res.get("cap"):
        return False
    if res.get("backend") == "sdpb":
        return res["bound"]
    return res.get("status") == "Solved"                                              # exploratory


def is_point(res: dict) -> bool:
    """A feasible point: every block at the returned moments is PSD to 1e3 thr, from the exact data."""
    if res.get("backend") == "sdpb":
        eigs = res.get("min_eigs") or []
        return bool(res.get("point")) and bool(eigs) and min(float(e) for e in eigs) >= -1e3 * res["threshold"]
    return res.get("status") == "Solved" and min(res.get("min_eigs", [-1])) >= -1e-7     # exploratory


def geometry(results: list, plane: str, lower: list = ()) -> dict:
    """Outer polygon, inner hull and their distance, in the window-normalized plane.

    The hierarchy is nested: every row and Ward identity of a level is present at all higher
    levels, so every bound proven at a lower level holds here too.  The outer set uses, in every
    direction, the strongest bound of this level and of `lower`; it can therefore never be weaker
    than at a lower level.  A feasible point that violates any bound, of this level or a lower one,
    is reported ("violation") and keeps the plane from converging.

    The error of an outer vertex, window corners included, is its distance to the inner set.  A
    vertex beyond TOLERANCE asks for the solves in `next`: its inherited directions at this level,
    a capped solve for each of its lines (or window edges) that has a bound but no feasible point
    (an infimum that is not attained), and the next untried dyadic direction between its two
    half-planes."""
    X, Y, (x0, x1), (y0, y1), (lo, hi) = PLANES[plane]
    dx, dy = x1 - x0, y1 - y0
    mine = [r for r in results if r["plane"] == plane]
    inh = [r for r in lower if r["plane"] == plane]
    best = {}
    for own, rs in ((True, mine), (False, inh)):
        for r in rs:
            if r.get("phi") is not None and in_range(r["phi"], plane) and is_bound(r):
                k = round(r["phi"], 9)
                if k not in best or float(r["value"]) > float(best[k][1]["value"]):
                    best[k] = (own, r)
    lines = list(best.values())
    # inner set: feasible points within one window size of the window.  Points far out along the
    # unbounded directions are feasible too, but at 1e20 window units they ruin the hull's precision;
    # dropping them only shrinks the inner set.
    feasible = [r for r in mine if is_point(r)]
    pts = np.array([(r["point"][X] / dx, r["point"][Y] / dy) for r in feasible
                    if x0 - dx <= r["point"][X] <= x1 + dx and y0 - dy <= r["point"][Y] <= y1 + dy]).reshape(-1, 2)
    if len(lines) < 3:
        return {"plane": plane, "bounds": sum(o for o, _ in lines), "inherited": sum(not o for o, _ in lines),
                "points": len(pts), "ok": False}

    def side(r, u, v):                               # >= 0 inside the half-plane of solve r
        return np.cos(r["phi"]) * u + np.sin(r["phi"]) * v - float(r["value"])

    # consistency: no feasible point may violate any bound, of this level or a lower one (checked on the
    # points within one window size, like the hull: far out, the certificates lose their precision)
    worst = min((side(b, u, v) for u, v in pts for _, b in lines), default=0.0)
    # half-planes  cos u + sin v >= value  ->  -cos u - sin v + value <= 0, then the window box
    window = [((-1, 0, x0 / dx), 0.0, (0, x0)), ((1, 0, -x1 / dx), np.pi, (0, x1)),
              ((0, -1, y0 / dy), np.pi / 2, (1, y0)), ((0, 1, -y1 / dy), -np.pi / 2, (1, y1))]
    hs = [(-np.cos(r["phi"]), -np.sin(r["phi"]), float(r["value"])) for _, r in lines] + [w[0] for w in window]
    outer, facets = _polygon(np.array(hs, dtype=float))
    hull = None
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
        except QhullError:                           # degenerate point sets (all at the tip) need joggling
            hull = ConvexHull(pts, qhull_options="QJ")
    tried = {_key(plane, spec_of(r)) for r in mine}

    def distance(v):                                 # Euclidean distance from v to the inner hull
        if (hull.equations[:, :2] @ v + hull.equations[:, 2]).max() <= 0:
            return 0.0
        P = pts[hull.vertices]
        E = np.roll(P, -1, axis=0) - P
        t = np.clip(((v - P) * E).sum(1) / (E * E).sum(1), 0, 1)
        return float(np.linalg.norm(v - P - t[:, None] * E, axis=1).min())

    def capped(base):                                # the next cap for a solve without a feasible point
        if any(is_point(r) and r.get("cap") and spec_of(r)[1] == base for r in mine):
            nxt = [c for c in CAPS if _key(plane, ("cap", base, c)) not in tried]
        else:
            nxt = [] if any(_key(plane, ("cap", base, c)) in tried for c in CAPS) else list(CAPS)
        return [("cap", base, nxt[0])] if nxt else []

    gap, nxt = 0.0, []
    for v, f in zip(outer, facets):
        d = distance(v) if hull else np.inf
        gap = max(gap, d)
        if d <= TOLERANCE:
            continue
        for i in f:
            if i < len(lines):
                own, r = lines[i]
                if not own and _key(plane, float(r["phi"])) not in tried:
                    nxt.append(float(r["phi"]))
                elif own and not is_point(r):
                    nxt += capped(float(r["phi"]))
            else:
                ax, val = window[i - len(lines)][2]
                for r in mine:
                    if (r.get("edge") and not r.get("cap") and _spec(r["edge"])[1:3] == (ax, val)
                            and is_bound(r) and not is_point(r)):
                        nxt += capped(_spec(r["edge"]))
        if all(i >= len(lines) for i in f[:2]):         # a window corner: is the corner itself feasible?
            (ax1, v1), (ax2, v2) = (window[i - len(lines)][2] for i in f[:2])
            if ax1 != ax2:
                xy = (v1, v2) if ax1 == 0 else (v2, v1)
                if _key(plane, ("corner", *xy)) not in tried:
                    nxt.append(("corner", float(xy[0]), float(xy[1])))
        ia, ib = sorted(f[:2], key=lambda i: i >= len(lines))   # a support line first
        if ia < len(lines):                          # directions between it and the other half-plane,
            a = lines[ia][1]["phi"]                  # turning the shorter way, within the scanned range
            b = lines[ib][1]["phi"] if ib < len(lines) else window[ib - len(lines)][1]
            b = min(max(a + (b - a + np.pi) % (2 * np.pi) - np.pi, lo), hi)
            new = next((a + (b - a) * j / 2 ** k for k in range(1, 6) for j in range(1, 2 ** k, 2)
                        if _key(plane, float(a + (b - a) * j / 2 ** k)) not in tried), None)
            if new is not None and abs(b - a) > 1e-9:
                nxt.append(float(new))
    return {"plane": plane, "X": X, "Y": Y, "window": [[x0, x1], [y0, y1]],
            "bounds": sum(o for o, _ in lines), "inherited": sum(not o for o, _ in lines), "points": len(pts),
            "outer": (outer * [dx, dy]).tolist(), "inner": (pts[hull.vertices] * [dx, dy]).tolist() if hull else [],
            "gap": float(gap) if np.isfinite(gap) else None, "violation": float(max(0.0, -worst)),
            "ok": bool(gap <= TOLERANCE and -worst <= CONSISTENCY), "next": list(dict.fromkeys(nxt))}


def _polygon(hs: np.ndarray):
    """Vertices (counterclockwise) and defining half-planes of {x : A x + b <= 0}."""
    norm = np.linalg.norm(hs[:, :2], axis=1)
    lp = linprog([0, 0, -1], A_ub=np.c_[hs[:, :2], norm], b_ub=-hs[:, 2], bounds=[(None, None)] * 3)
    inside = lp.x[:2]                                # Chebyshev center
    his = HalfspaceIntersection(hs, inside)
    order = np.argsort(np.arctan2(*(his.intersections - inside).T[::-1]))
    return his.intersections[order], [his.dual_facets[i] for i in order]


def _finite(x):
    """Strict JSON: non-finite numbers (Clarabel reports NaN objectives when infeasible) become null."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def write(run: Path, results: list, lower: list = ()) -> dict:
    """Write the solves and the geometry of the scanned planes to regions.json, keeping other planes."""
    planes = sorted({r["plane"] for r in results})
    path = run / "regions.json"
    old = json.loads(path.read_text()) if path.exists() else {"solves": [], "regions": {}}
    doc = {"tolerance": TOLERANCE, "solves": [r for r in old["solves"] if r["plane"] not in planes] + results,
           "regions": {**{pl: g for pl, g in old["regions"].items() if pl not in planes},
                       **{pl: geometry(results, pl, lower) for pl in planes}}}
    path.write_text(json.dumps(_finite(doc), indent=1, default=str, allow_nan=False))
    return doc
