"""Pointwise supersymmetry residual of a state on the gauge slice, and which new seeds could absorb it.

On the slice X = D(x) a singlet is an F_*-valued function Psi(x) = psi(D(x)) = sum_k phi_k(x) V_k(x): V_k is the
slice value of core k (Y^p Lambda_sigma) and phi_k the sum of c_j f_j times the invariant prefactor over its trial
wavefunctions.  There

    H Psi   = -1/2 w^-1 sum_a d_a (w d_a Psi) + T_ang Psi + u Psi + Y(x) Psi,  w = prod_{a<b} (x_a^2 - x_b^2) prod_a x_a^6,
    T_ang   = -1/2 sum_a RR_a / x_a^2 - 1/2 sum_{a<b} (d_ab^2 + d_ba^2),
    H_aux   = (1/16) sum_alpha Q_alpha R^2 Q_alpha = s H - frakB/8 - X.d - (i frakV)/8,

with RR_a = sum_{i>3} R_ai^2 and d_ab, d_ba = (x_b G_ab + x_a R_ab, x_a G_ab + x_b R_ab) / (x_b^2 - x_a^2) the rotation
generators that give the 24 derivatives off the slice (the paper's Eq. (E.6); contract.PointEvaluator), frakB =
sum X_ai B_{ai,bk} d_bk and i frakV = sum_e x^e V_e.  Integrated over the slice (int d^27X F = pi^{27/2} I[F] / I[e^-s],
radial_tail) these give the norm, E and <H_aux> independently of the analytic blocks.

For an optimum the residual g = (H_aux - <H_aux>) Psi is orthogonal to the trial space; at each x the part of g(x)
outside span{V_k(x)} is out of reach of every trial wavefunction over these cores, while the part inside could be
removed by better coefficient functions.  `split` integrates both by radius; `rank` chooses, greedily, the candidate
seeds whose towers {Y^p Lambda_sigma, p <= pmax} absorb most of the unreachable part (block Schur complements, batched
over points).

    python analysis/slice_residual.py check STATE                      # norm, <H>, <H_aux> against the stored values
    python analysis/slice_residual.py split STATE [--extra SEEDS.json --pmax 8]
    python analysis/slice_residual.py rank STATE CANDIDATES.json [--pmax 4 --select 20]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from bfss.algebra import invariants
from bfss.contract import OPERATORS, OperatorLibrary
from bfss.covariants import SeedSet
from bfss.fiber import PLANE_CHARACTER, TRIVIAL, ComplexContext
from bfss.state import load
from radial_tail import Coefficients, jacobian_constant, profile, slice_measure

def invariant_jet(exps, g):
    """s^k u^l v^m, (k, l, m) = exps, at squared singular values g, with its g-gradient and g-Hessian."""
    k, l, m = exps
    s, u, v = invariants(g)
    pw = lambda x, n, d: 0.0 if n < d else float(np.prod(range(n - d + 1, n + 1))) * x ** (n - d)
    F = {(a, b, c): pw(s, k, a) * pw(u, l, b) * pw(v, m, c) for a in range(3) for b in range(3) for c in range(3)}
    off = 1 - np.eye(3)
    d = [np.ones(3), s - g, np.array([g[1] * g[2], g[0] * g[2], g[0] * g[1]])]  # gradients of s, u, v
    h = [np.zeros((3, 3)), off, off * np.array([[g[2], g[2], g[1]], [g[2], g[2], g[0]], [g[1], g[0], g[0]]])]
    unit = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    grad = sum(F[unit[i]] * d[i] for i in range(3))
    hess = sum(F[tuple(np.add(unit[i], unit[j]))] * np.outer(d[i], d[j]) for i in range(3) for j in range(3))
    return F[0, 0, 0], grad, hess + sum(F[unit[i]] * h[i] for i in range(3))


class SliceOperators:
    """The slice operators as sparse matrices in orthonormal K_4-sector coordinates (sqrt(weight) v[rep])."""

    def __init__(self, library: OperatorLibrary, seeds_json):
        self.ctx = ComplexContext(library.ops, library.space)
        self.sw = {ch: np.sqrt(s.weight.astype(float)) for ch, s in self.ctx.sectors.items()}
        self.seeds = SeedSet.from_json(seeds_json)
        self.values = dict(zip([s.name for s in self.seeds.seeds], self.seeds.evaluate(self.ctx)))
        ops = self.ctx.operators
        self.K = [self.op(f"K{a}{a}") for a in range(3)]
        self.RR = [self.op(f"RR{a}") for a in range(3)]
        self.Bslice = {(a, b): self.op(f"B{a}{b}{b}") for a in range(3) for b in range(3) if f"B{a}{b}{b}" in ops}
        self.BR = {(a, b): self.op(f"BR{a}{b}") for a in range(3) for b in range(3)}
        self.cV = {tuple(map(int, n[2:])): self.op(n) for n in ops if n.startswith("cV")}
        self.planes = {(b, k): [(self.op(m, TRIVIAL, ch), self.op(m, ch, TRIVIAL)) for m in (f"G{b}{k}", f"R{b}{k}")]
                       for (b, k), ch in PLANE_CHARACTER.items()}
        self.Bplane = {(a, i, j): self.op(f"B{a}{i}{j}", ch, TRIVIAL) for (b, k), ch in PLANE_CHARACTER.items()
                       for i, j in ((b, k), (k, b)) for a in range(3) if f"B{a}{i}{j}" in ops}

    def op(self, name, src=TRIVIAL, dst=TRIVIAL):
        return (sp.diags(self.sw[dst]) @ self.ctx.restricted(name, src, dst) @ sp.diags(1 / self.sw[src])).tocsr()

    def polynomial(self, name):
        """Slice polynomial of a seed: exponents (n, 3) and coefficient vectors (n, dim), in trivial-sector
        coordinates.  The values at the representatives are taken as they are: for some published seeds (A, C,
        H0-H3) single terms of the complex realization are not sector vectors, but their sum at every x is (its
        norm agrees with the full orbit-space density of radial_tail to 1e-11)."""
        terms, rep = self.values[name].terms, self.ctx.sectors[TRIVIAL].rep
        return (np.array(list(terms), dtype=int),
                np.stack([self.sw[TRIVIAL] * np.asarray(v, dtype=complex)[rep] for v in terms.values()]))

    def yukawa(self, x):
        return sum(xa * Ka for xa, Ka in zip(x, self.K))

    def off_slice(self, x, Psi):
        """The derivatives d_bk Psi, d_kb Psi off the slice, for each plane (b, k)."""
        g, d = x**2, {}
        for (b, k), ((Gf, _), (Rf, _)) in self.planes.items():
            G, R = Gf @ Psi, Rf @ Psi
            d[b, k], d[k, b] = (x[k] * G + x[b] * R) / (g[k] - g[b]), (x[b] * G + x[k] * R) / (g[k] - g[b])
        return d

    def hamiltonian(self, x, Psi, dPsi, d2Psi, d):
        """H Psi, given the derivatives d off the slice."""
        g = x**2
        rad = d2Psi.sum(0) + sum((6 / x[a] + sum(2 * x[a] / (g[a] - g[b]) for b in range(3) if b != a)) * dPsi[a]
                                 for a in range(3))
        out = -rad / 2 + invariants(g)[1] * Psi + self.yukawa(x) @ Psi
        out = out - sum(self.RR[a] @ Psi / g[a] for a in range(3)) / 2
        for (b, k), ((_, Gb), (_, Rb)) in self.planes.items():
            for (i, j), (cg, cr) in (((b, k), (x[k], x[b])), ((k, b), (x[b], x[k]))):
                out = out - (cg * (Gb @ d[i, j]) + cr * (Rb @ d[i, j])) / (g[k] - g[b]) / 2
        return out

    def weighted_residual(self, x, Psi, dPsi, d2Psi):
        """H_aux Psi = s H Psi - frakB Psi / 8 - X.d Psi - (i frakV) Psi / 8 (and H Psi)."""
        d = self.off_slice(x, Psi)
        H = self.hamiltonian(x, Psi, dPsi, d2Psi, d)
        frakB = sum(x[a] * (B @ dPsi[b]) for (a, b), B in self.Bslice.items())
        frakB = frakB - sum(x[a] / x[b] * (BR @ Psi) for (a, b), BR in self.BR.items())
        frakB = frakB + sum(x[a] * (M @ d[i, j]) for (a, i, j), M in self.Bplane.items())
        iV = sum(np.prod(x ** np.array(e)) * (M @ Psi) for e, M in self.cV.items())
        return (x**2).sum() * H - frakB / 8 - sum(x[a] * dPsi[a] for a in range(3)) - iV / 8, H


class SliceState:
    """A state on the slice: Psi, its first and second slice derivatives, and the core values at x."""

    def __init__(self, state, ops: SliceOperators):
        self.ops, self.coef = ops, Coefficients(state)
        self.cores, self.norm = self.coef.cores, float(state.basis.seeds.norm)
        self.seeds = {s: ops.polynomial(state.basis.seeds.seeds[s].name) for s in {c[0] for c in self.cores}}
        self.top = max(p for _, p in self.cores)

    def coefficients(self, x):
        """phi_k(x) with its first and second x-derivatives."""
        g, cf = x**2, self.coef
        P = [[q[..., 0] for q in profile(a, b, g[:, None], jet=True)] for a, b in cf.widths]
        I = [invariant_jet(e, g) for e in cf.exps]
        f, fg, fh = (np.array([p[i] for p in P])[cf.w] for i in range(3))
        o, og, oh = (np.array([q[i] for q in I])[cf.e] for i in range(3))
        F, Fg = f * o, fg * o[:, None] + f[:, None] * og
        Fh = fh * o[:, None, None] + fg[:, :, None] * og[:, None, :] + og[:, :, None] * fg[:, None, :] + f[:, None, None] * oh
        dx, hx = 2 * x * Fg, 4 * np.outer(x, x) * Fh + 2 * Fg[:, :, None] * np.eye(3)
        sum_k = lambda v: np.bincount(cf.k, weights=cf.c * v, minlength=len(self.cores))
        return (sum_k(F), np.stack([sum_k(dx[:, a]) for a in range(3)], 1),
                np.stack([[sum_k(hx[:, a, b]) for b in range(3)] for a in range(3)]).transpose(2, 0, 1))

    def at(self, x):
        """Psi, d_a Psi (3, dim), d_a^2 Psi (3, dim) and the core values (K, dim), the cores by the recursion
        Y^p Lambda = Y (Y^(p-1) Lambda) with d_a (Y V) = K_a V + Y d_a V and d_a^2 (Y V) = 2 K_a d_a V + Y d_a^2 V."""
        phi, dphi, hphi = self.coefficients(x)
        Y, K = self.ops.yukawa(x), self.ops.K
        seeds = sorted(self.seeds)
        cur = np.stack([jet(*self.seeds[s], x) for s in seeds])  # (seeds, 7, dim): V, d_a V, d_a^2 V
        index = {c: i for i, c in enumerate(self.cores)}
        V, dV, d2V = (np.zeros((n, len(self.cores), cur.shape[-1]), complex) for n in (1, 3, 3))
        for p in range(self.top + 1):
            for i, s in enumerate(seeds):
                if (s, p) in index:
                    j = index[s, p]
                    V[0, j], dV[:, j], d2V[:, j] = cur[i, 0], cur[i, 1:4], cur[i, 4:]
            nxt = np.stack([(Y @ cur[:, j].T).T for j in range(7)], 1)
            for a in range(3):
                nxt[:, 1 + a] += (K[a] @ cur[:, 0].T).T
                nxt[:, 4 + a] += 2 * (K[a] @ cur[:, 1 + a].T).T
            cur = nxt
        V = V[0]
        dPsi = np.stack([dphi[:, a] @ V + phi @ dV[a] for a in range(3)])
        d2Psi = np.stack([hphi[:, a, a] @ V + 2 * dphi[:, a] @ dV[a] + phi @ d2V[a] for a in range(3)])
        return phi @ V, dPsi, d2Psi, V


def jet(mons, vecs, x):
    """A slice polynomial and its first and second derivatives along each axis: (7, dim)."""
    xm = np.prod(x ** mons, axis=1)
    rows = [xm @ vecs]
    for order in (1, 2):
        for a in range(3):
            m = mons[:, a]
            c = m * xm / x[a] if order == 1 else m * (m - 1) * xm / x[a] ** 2
            rows.append(np.where(m >= order, c, 0.0) @ vecs)
    return np.stack(rows)


def slice_rule(rmax, nr, nshape):
    """Gauss rule over 0 < x3 < x2 < x1 < rmax: points (n, 3) and weights including w(x)."""
    t, w = np.polynomial.legendre.leggauss(nr)
    s, ws = np.polynomial.legendre.leggauss(nshape)
    r, wr, s, ws = (t + 1) * rmax / 2, w * rmax / 2, (s + 1) / 2, ws / 2
    x1, t2, t3 = (a.ravel() for a in np.meshgrid(r, s, s, indexing="ij"))
    W = (wr[:, None, None] * ws[None, :, None] * ws[None, None, :]).ravel()
    x2 = x1 * t2
    x3 = x2 * t3
    return np.stack([x1, x2, x3], 1), W * x1 * x2 * slice_measure(x1, x2, x3)


def projector(V):
    """Orthonormal basis of the span of the (column-normalized) core values: pivoted QR."""
    Q, R, _ = scipy.linalg.qr((V / np.linalg.norm(V, axis=1, keepdims=True)).T, mode="economic", pivoting=True)
    return Q[:, np.abs(np.diag(R)) > 1e-12]


def candidate_towers(ops, names, pmax, x):
    """Values Y^p Lambda_sigma (p <= pmax) of candidate seeds at x, as normalized columns."""
    Y, cols = ops.yukawa(x), []
    for n in names:
        v = jet(*ops.polynomial(n), x)[0]
        for _ in range(pmax + 1):
            cols.append(v)
            v = Y @ v
    T = np.stack(cols, axis=1)
    return T / np.linalg.norm(T, axis=0, keepdims=True)


_RUN = {}


def _point(i):
    """Per-point data for split (radius bin and integrands) and rank (Gram blocks of the candidate towers)."""
    ss, ops, eps, extra, pmax = (_RUN[k] for k in ("state", "ops", "eps", "extra", "pmax"))
    x = _RUN["points"][i]
    Psi, dPsi, d2Psi, V = ss.at(x)
    HauxPsi, HPsi = ops.weighted_residual(x, Psi, dPsi, d2Psi)
    g = HauxPsi - eps * Psi
    Q = projector(V)
    perp = g - Q @ (Q.conj().T @ g)
    out = {"M": np.vdot(Psi, Psi).real, "H": np.vdot(Psi, HPsi).real, "Haux": np.vdot(Psi, HauxPsi).real,
           "g2": np.vdot(g, g).real, "perp": np.vdot(perp, perp).real}
    if extra:
        T = candidate_towers(ops, extra, pmax, x)
        T = T - Q @ (Q.conj().T @ T)
        out["G"], out["b"] = T.conj().T @ T, T.conj().T @ perp
    return out


def _one_blas_thread():
    """A worker process per CPU: each forked worker runs its BLAS on one thread instead of the parent's pool."""
    threadpool_limits(1)


def run_points(state_path, rule, cache, extra=(), pmax=0, candidates=()):
    """Evaluate all points of a slice rule in forked workers, one per CPU, with the operator library of the cache
    directory; returns (weights, per-point records, constant)."""
    state = load(state_path)
    seeds_json = state.basis.seeds.to_json()
    seeds_json = dict(seeds_json, seeds=seeds_json["seeds"] + list(candidates))
    ops = SliceOperators(OperatorLibrary(Path(cache) / OPERATORS), seeds_json)
    eps = float(state.metadata.get("Haux_eigenvalue", 0.0))  # the optimizer's eigenvalue epsilon
    points, weights = slice_rule(*rule)
    _RUN.update(state=SliceState(state, ops), ops=ops, eps=eps, extra=list(extra), pmax=pmax, points=points)
    with mp.get_context("fork").Pool(initializer=_one_blas_thread) as pool:
        data = pool.map(_point, range(len(points)), chunksize=8)
    return points, weights, data, jacobian_constant() * _RUN["state"].norm


def greedy(W, G, B, L, select, names):
    """Block greedy choice: P is the Gram of the candidate towers projected away from the chosen ones and r their
    projected overlaps with the residual; candidate c absorbs r_c^H P_cc^+ r_c; choosing c* downdates P and r."""
    P, r, chosen, total = G.copy(), B.copy(), [], 0.0
    blocks = [np.arange(c * L, (c + 1) * L) for c in range(G.shape[1] // L)]

    def pinv(M):
        ev, U = np.linalg.eigh(M)
        keep = ev > 1e-10 * np.maximum(ev.max(axis=1, keepdims=True), 1e-300)
        return np.einsum("pij,pj,pkj->pik", U, np.where(keep, 1 / np.where(keep, ev, 1), 0), U.conj())

    for _ in range(select):
        gains = [-1.0 if c in chosen else float(np.sum(W * np.einsum(
            "pi,pij,pj->p", r[:, bc].conj(), pinv(P[:, bc][:, :, bc]), r[:, bc]).real)) for c, bc in enumerate(blocks)]
        best = int(np.argmax(gains))
        chosen.append(best)
        total += gains[best]
        bc = blocks[best]
        col, Pi = P[:, :, bc], pinv(P[:, bc][:, :, bc])
        r = r - np.einsum("pnl,plm,pm->pn", col, Pi, r[:, bc])
        P = P - np.einsum("pnl,plm,pkm->pnk", col, Pi, col.conj())
        yield names[best], total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="directory of the operator library")
    parser.add_argument("--rule", default="10,40,20", help="rmax,nr,nshape of the slice rule (physical radii)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="norm, <H> and <H_aux> by slice quadrature").add_argument("state", type=Path)
    spl = sub.add_parser("split", help="residual inside and outside the span of the cores, by radius")
    spl.add_argument("state", type=Path)
    spl.add_argument("--extra", type=Path, help="JSON list of seed definitions whose towers are added")
    spl.add_argument("--pmax", type=int, default=8)
    rnk = sub.add_parser("rank", help="greedy choice of candidate seeds by the residual they absorb")
    rnk.add_argument("state", type=Path)
    rnk.add_argument("candidates", type=Path)
    rnk.add_argument("--pmax", type=int, default=4)
    rnk.add_argument("--select", type=int, default=20)
    args = parser.parse_args(argv)
    rule = tuple(float(v) if i == 0 else int(v) for i, v in enumerate(args.rule.split(",")))
    start = time.time()
    cands = json.loads(args.candidates.read_text()) if args.command == "rank" else (
        json.loads(args.extra.read_text()) if args.command == "split" and args.extra else [])
    names = [c["name"] for c in cands]
    pmax = args.pmax if args.command != "check" else 0
    points, W, data, K0 = run_points(args.state, rule, args.cache, names, pmax, cands)
    total = lambda key, sel=range(len(data)): K0 * sum(W[i] * data[i][key] for i in sel)
    M = total("M")
    print(f"norm {M:.10f}  <H> = {total('H') / M:.10e}  <Haux> = {total('Haux') / M:.10e}  ({len(points)} points, "
          f"{time.time() - start:.0f}s)", flush=True)
    if args.command == "check":
        meta = load(args.state).metadata.get("observables", {})
        print("stored:", {k: meta.get(k) for k in ("M", "E", "Haux")}, flush=True)
        return
    g2, perp = total("g2"), total("perp")
    print(f"int |g|^2 = {g2:.4e}; outside the span of the cores: {perp:.4e} ({perp / g2:.1%})", flush=True)
    if args.command == "split":
        radius = np.linalg.norm(points, axis=1)
        for lo in np.arange(0, rule[0], 0.5):
            sel = [i for i in range(len(points)) if lo <= radius[i] < lo + 0.5]
            if sel:
                print(f"  |x| in [{lo:.1f},{lo + 0.5:.1f}): probability {total('M', sel) / M:.4f}  "
                      f"|g|^2 {total('g2', sel):.3e}  outside {total('perp', sel):.3e}")
    if not names:
        return
    G, B = np.stack([d["G"] for d in data]), np.stack([d["b"] for d in data])
    steps = list(enumerate(greedy(K0 * W, G, B, pmax + 1, args.select if args.command == "rank" else len(names), names), 1))
    for step, (name, absorbed) in steps if args.command == "rank" else steps[-1:]:
        print(f"{step:3d}. {name:20s} absorbs {absorbed:.4e} in all ({absorbed / perp:.1%} of the unreachable residual)",
              flush=True)


if __name__ == "__main__":
    main()
