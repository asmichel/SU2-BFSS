"""The level-L hierarchy over the complete basis.

Rows: every multiplet of level <= L with finite norm (basis.py), one Gram block per irrep.  Ward
identities for N in {H, gauss, J, Q} (H|Omega> = 0 with E = 0, gauge and rotation singlet, Q|Omega> = 0): for X a
row with l_X + l_N <= L, or a singlet in the positions alone, every multiplet n in X * N annihilates the
ground state, so <r^+ n> = 0 for every row r of the irrep of n (n: the Clebsch-Gordan highest-weight
vectors of X * N).  A Ward form is admitted when the part of <r^+ n> that is not fixed by symmetry lies
in the moments of level <= 2L (l_r + l_X + SHIFT[N] <= 2L) and the expectation is finite in the true
ground state (growth exponents e_r + e_X + e_N < 9).  Rows annihilating the ground state (N and the
products X * N) are found by the reduction as common kernels of the blocks (facial reduction).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass

from . import basis as B
from . import model as M
from . import pairing as Pa

TAIL_FORM = 18                     # valley degree 2e of a finite expectation
WARD = {"H": (lambda: (M.H(),), (0, 0, 0, 0, 0), 2), "gauss": (M.gauss, (2, 0, 0, 0, 0), 6),
        "J": (M.J_rot, (0, 2, 2, 0, 0), 6), "Q": (M.Q, (0, 1, 1, 1, 1), 1)}
OBSERVABLES = {"R2": M.R2, "R4": M.R4, "u": M.u, "T": M.T, "Y": M.Y, "D": M.D}
SINGLET = (0, 0, 0, 0, 0)


@dataclass
class Row:
    lam: tuple
    grade: tuple
    index: int
    poly: dict                     # exact highest-weight vector
    tail: int                      # valley degree 2e

    @property
    def level(self) -> int:
        return B.grade_level(self.grade)

    @property
    def name(self) -> str:
        return "x%dp%dt%d" % self.grade + (f".{self.index}" if self.index else "")


@dataclass
class Hierarchy:
    level: int
    rows: dict                     # lam -> [Row]
    ward: list                     # [(N, lam, label, row index, index of the null multiplet)]
    nulls: list                    # [(lam, (rational, sqrt2) hw vectors)] in X * N
    special: dict                  # N or observable -> (rational, sqrt2) hw vectors

    def manifest(self) -> dict:
        return {"level": self.level,
                "blocks": {family(lam): [r.name for r in rs] for lam, rs in self.rows.items()},
                "tails": {family(lam): [r.tail / 2 for r in rs] for lam, rs in self.rows.items()},
                "ward": [w[2] for w in self.ward]}


def family(lam: tuple) -> str:
    return "j%g [%s]" % (lam[0] / 2, ",".join("%g" % (s / 2) for s in lam[1:]))


def _sector(item):
    (g, lam), m = item
    t = time.time()
    return B.sector(g, lam, m), round(time.time() - t, 1)


def growth(poly: dict) -> int:
    """Valley degree (2e) of a weight-coordinate polynomial: the largest over its grades."""
    by = {}
    for m, z in poly.items():
        by.setdefault((len(m[0]), len(m[1]), len(m[2])), {})[m] = z
    return max(B.valley_degrees(list(part), g, [list(part.values())], 1)[0] for g, part in by.items())


def highest(polys: list, lam: tuple) -> tuple:
    """(rational, sqrt2) parts of the highest-weight vector of a multiplet given by Cartesian components."""
    act = B.action()
    for P in polys:
        parts = []
        for part in B.from_cartesian(P):
            w = {m: z for m, z in part.items() if _weight(m, act) == lam}
            parts.append(w)
        if parts[0] or parts[1]:
            return tuple(parts)
    raise ValueError(f"no component of weight {lam}")


def _weight(m, act) -> tuple:
    w = [0] * 5
    for kind, idx in zip("xpf", m):
        for k in idx:
            w = [a + b for a, b in zip(w, act[kind]["wt"][k])]
    return tuple(w)


def canonical(level: int, workers: int = 1, log=print) -> Hierarchy:
    """The hierarchy of a level: rows of every irrep, null multiplets and Ward forms, special multiplets."""
    B.action()
    items = sorted(B.multiplicities(level).items(), key=lambda kv: -len(B.monomials(*kv[0])))
    rows = {}
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    for sec, seconds in (pool.imap_unordered(_sector, items) if pool else map(_sector, items)):
        log(json.dumps({"event": "sector", "grade": sec.grade, "lam": sec.lam, "monomials": len(sec.mons),
                        "multiplets": len(sec.hw), "rows": len(sec.rows), "seconds": seconds}))
        for i in range(len(sec.rows)):
            rows.setdefault(sec.lam, []).append(Row(sec.lam, sec.grade, i, sec.poly(i), sec.tails[i]))
    if pool:
        pool.close()
    rows = {lam: sorted(rs, key=lambda r: (r.level, r.grade, r.index)) for lam, rs in
            sorted(rows.items(), key=lambda kv: (B.dimension(kv[0]), kv[0]))}
    special = {name: highest(f(), lam) for name, (f, lam, _) in WARD.items()}
    special |= {name: highest([f()], SINGLET) for name, f in OBSERVABLES.items()}
    special["P2"] = highest([M.T().scale(num=2)], SINGLET)
    xonly = [r for r in rows[SINGLET] if r.grade[1] == r.grade[2] == 0]
    irreps, ward, nulls = {}, [], []

    def irrep(lam, ref):
        if lam not in irreps:
            irreps[lam] = Pa.Irrep(lam, ref)
        return irreps[lam]
    for name, (f, nu, shift) in WARD.items():
        parts = special[name]
        irrN = irrep(nu, parts[0] or parts[1])
        compsN = [irrN.components(part) if part else None for part in parts]
        RN = irrN.raising(next(c for c in compsN if c))
        lN, tN = max(B.grade_level((len(m[0]), len(m[1]), len(m[2]))) for part in parts for m in part), \
            max(growth(part) for part in parts if part)
        Xs = [X for rs in rows.values() for X in rs if X.level + lN <= level] + \
             [X for X in xonly if X.level + lN > level]
        for X in Xs:
            irrX = irrep(X.lam, X.poly)
            compsX = irrX.components(X.poly)
            RX = irrX.raising(compsX)
            for lam, rs in rows.items():
                for a in Pa.tensor_highest(irrX, RX, irrN, RN, lam):
                    n = tuple(Pa._lin((z, Pa.star(compsX[k], cN[d])) for (k, d), z in a.items()) if cN else {}
                              for cN in compsN)
                    if not (n[0] or n[1]):
                        continue
                    nulls.append((lam, n))
                    for i, r in enumerate(rs):
                        if r.level + X.level + shift <= 2 * level and r.tail + X.tail + tN < TAIL_FORM:
                            ward.append((name, lam, f"<{r.name}^+ ({X.name} {family(X.lam)} * {name})_{len(nulls) - 1}>",
                                         i, len(nulls) - 1))
    return Hierarchy(level, rows, ward, nulls, special)
