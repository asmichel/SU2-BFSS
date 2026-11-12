"""Radial density p(r) of a state along the commuting valley and the fits of its r^-10 tail.

p(r) = int d^27X |psi(X)|^2 delta(r - x_1(X)) is the density of the largest singular value r = x_1 of X,
which the asymptotic analysis predicts to fall as p(r) ~ C r^-10 with C = 1890 sqrt(2).  On the gauge
slice X = D(x), x1 > x2 > x3 > 0,

    psi(D(x)) = sum_j c_j f_{a_j,b_j}(x) s^k u^l v^m Y(x)^p Lambda_sigma(x),

and an invariant F integrates as int d^27X F = pi^{27/2} I[F] / I[e^{-s}] (the slice measure of the paper's
Eq. (E.2)), with I[h] = int_{x1>x2>x3>0} Delta(x^2) (x1 x2 x3)^6 h dx; p(r) is the shape integral over x2, x3 at x1 = r.
psi(D(x)) is assembled as a vector in orbit coordinates, so that its norm is a sum of squares: squaring
first (phi^T G phi with the core Gram family) cancels by ~1e14 for a nearly dependent basis.

    python analysis/radial_tail.py density STATE OUT [--radii 3:8:0.025] [--orders 32,48,64] [--shard i/n]
    python analysis/radial_tail.py norm STATE OUT     # norm, <r^2>, rms r, <rho^2>, and <R^2>
    python analysis/radial_tail.py fit OUT SHARD.json ... [--norm NORM.json] [--compare r0,r1] [--window r0,r1]

Grids are first:last:step (inclusive) or comma-separated.  OUT of "density" may contain {task}, replaced by
the shard index; the shard defaults to 0/1, every radius.  The fits work on a cubic spline of ln(r^10 p) against ln r: on a window
[r0, ratio r0], C_fit is the coefficient at fixed power 10 (least squares in ln p), the free power is the
least-squares slope, and the scatter is the rms of r^10 p / C_fit - 1; the flattest window of each ratio
(smallest spread of ln(r^10 p)) is chosen from the shape alone, never from the value of C.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import i0e, ive

from bfss.algebra import invariants
from bfss.contract import OPERATORS, OperatorLibrary
from bfss.fiber import ComplexContext
from bfss.hilbert import ProjectedGaussian, TrialFunction
from bfss.state import load

C_PRED = 1890 * np.sqrt(2)
RATIOS = (1.05, 1.1, 1.15, 1.2, 1.3, 1.5)


def unit_rule(order):
    """Gauss-Legendre nodes and weights on (0, 1)."""
    t, w = np.polynomial.legendre.leggauss(order)
    return (t + 1) / 2, w / 2


SPHERE = unit_rule(256)  # on cos(theta)


def profile(a, b, g, jet=False):
    """f_{a,b}(D(x)) for squared singular values g1 >= g2 >= g3 (columns of g), vectorized over points; with
    ``jet`` also its g-gradient (3, n) and g-Hessian (3, 3, n).

    The azimuthal average of exp(n^T diag(eta g) n) about the x3 axis is a Bessel function, so
    f = int_0^1 dc exp(-a s/2 + eta g3 c^2 + eta (1-c^2)(g1+g2)/2) I0(z), z = eta (1-c^2)(g1-g2)/2,
    eta = (a-b)/2.  I0 is even and i0e(z) = exp(-|z|) I0(z), so the exponent below carries |z|; it stays
    non-positive for either sign of eta, and nothing overflows far along the valley.  The exponent is linear
    in g with gradient l and z has gradient m, so d(e I0) = e (l I0 + m I1) and the Hessian adds
    (l m^T + m l^T) I1 + m m^T (I0 + I2)/2."""
    eta = (a - b) / 2
    g1, g2, g3 = g
    s = g1 + g2 + g3
    out, grad, hess = np.zeros(g.shape[1]), np.zeros((3, g.shape[1])), np.zeros((3, 3, g.shape[1]))
    for c, w in zip(*SPHERE):
        t = 1 - c * c
        z = t * eta * (g1 - g2) / 2
        e = w * np.exp(-a * s / 2 + eta * g3 * c * c + t * eta * (g1 + g2) / 2 + np.abs(z))
        i0 = i0e(z)
        out += e * i0
        if jet:
            l, m, i1 = np.array([eta * t / 2, eta * t / 2, eta * c * c]) - a / 2, np.array([1, -1, 0]) * eta * t / 2, ive(1, z)
            grad += e * (l[:, None] * i0 + m[:, None] * i1)
            hess += e * (np.outer(l, l)[..., None] * i0 + (np.outer(l, m) + np.outer(m, l))[..., None] * i1
                         + np.outer(m, m)[..., None] * (i0 + ive(2, z)) / 2)
    return (out, grad, hess) if jet else out


class Coefficients:
    """A state's functions as index arrays into its cores, its distinct widths and invariant exponents, with their
    coefficients: phi_k(x) = sum of c_j f_{w_j}(x) times the invariant monomial of exponents e_j, over the
    functions j of core k."""

    def __init__(self, state):
        basis = state.basis
        self.cores = basis.cores()
        widths = sorted({(f.profile.a, f.profile.b) for f in basis.functions})
        self.widths, self.exps = [(float(a), float(b)) for a, b in widths], sorted({f.invariant for f in basis.functions})
        index = lambda items: {item: i for i, item in enumerate(items)}
        k_of, w_of, e_of = index(self.cores), index(widths), index(self.exps)
        self.k = np.array([k_of[basis.core(f)] for f in basis.functions])
        self.w = np.array([w_of[f.profile.a, f.profile.b] for f in basis.functions])
        self.e = np.array([e_of[f.invariant] for f in basis.functions])
        self.c = np.array([float(c) for c in state.coefficients])

    def phi(self, prof, inv):
        """phi_k at points, from the profiles (widths x points) and the invariant monomials (exponents x points)."""
        out = np.zeros((len(self.cores), prof.shape[1]))
        for k, w, e, c in zip(self.k, self.w, self.e, self.c):
            out[k] += c * prof[w] * inv[e]
        return out


class Density:
    """|psi_c(D(x))|^2 of the canonical (largest b = 1) form psi_c of a state, vectorized over points.

    psi(D(x)) = sum_k phi_k(x) sum_m x^m v_km in orbit coordinates (weighted by the square roots of the
    orbit sizes), with phi_k the sum of c_j f_j times its invariant prefactor over the functions of core k and
    v_km the slice polynomial of the core Y^p Lambda_sigma.  The physical state is
    psi(X) = lam^{nu/2} psi_c(sqrt(lam) X)."""

    def __init__(self, state, library):
        lam = state.basis.reference_width()
        canonical = state.dilated(1 / lam)
        self.lam = float(lam)
        basis = canonical.basis
        self.norm = float(basis.seeds.norm)
        self.coef = Coefficients(canonical)
        cores = self.coef.cores
        self.K = len(cores)
        ctx = ComplexContext(library.ops, library.space)
        env = basis.seeds.environment(ctx)
        weight = np.sqrt(library.space.sizes.astype(float))
        cols, re, im = [], [], []  # per-core blocks, so that no full complex copy is ever held
        for k, (seed, power) in enumerate(cores):
            core = TrialFunction(ProjectedGaussian(1, 1), (0, 0, 0), power, basis.seeds.seeds[seed].name)
            terms = core.covariant(basis.seeds).on_slice(ctx, env).terms
            block = np.stack([np.asarray(v, dtype=complex) for v in terms.values()], axis=1) * weight[:, None]
            cols += [(k, *m) for m in terms]
            re.append(block.real.copy())
            im.append(block.imag.copy())
            del block, terms
        self.A = []
        for part in (re, im):
            if any(np.any(b) for b in part):
                self.A.append(np.concatenate(part, axis=1))
            part.clear()
        self.cols = np.array(cols)

    def __call__(self, x, chunk=512):
        g = x**2
        s, u, v = invariants(g)
        phi = self.coef.phi(np.array([profile(a, b, g) for a, b in self.coef.widths]),
                            np.array([s**k * u**l * v**m for k, l, m in self.coef.exps]))
        k, m = self.cols[:, 0], self.cols[:, 1:]
        out = np.empty(x.shape[1])
        for lo in range(0, x.shape[1], chunk):
            xs = x[:, lo:lo + chunk]
            W = phi[k, lo:lo + chunk] * np.prod(xs[None, :, :] ** m[:, :, None], axis=1)
            out[lo:lo + chunk] = sum(np.einsum("ip,ip->p", y, y) for y in (A @ W for A in self.A))
        return out * self.norm


def slice_measure(x1, x2, x3):
    """Delta(x^2) (x1 x2 x3)^6, the measure of the gauge slice x1 > x2 > x3 > 0 up to its constant."""
    g1, g2, g3 = x1**2, x2**2, x3**2
    return (g1 - g2) * (g1 - g3) * (g2 - g3) * (x1 * x2 * x3) ** 6


def jacobian_constant():
    """pi^{27/2} / I[e^{-s}] = 1024 pi^13 / 4725, the constant of the slice measure (the paper's Eq. (E.2))."""
    return 1024 * np.pi**13 / 4725


def shape_integrals(dens, radii, order, K0):
    """p(r) and the conditional moments <rho^2>_r, <rho^4>_r (rho^2 = sigma_2^2 + sigma_3^2) at physical radii, by
    an order x order Gauss rule over 0 < x3 < x2 < r on the gauge slice.  At the canonical radius sqrt(lam) r,
    p gains a factor sqrt(lam) and rho^2 a factor 1/lam."""
    sl = np.sqrt(dens.lam)
    t, w = unit_rule(order)
    T2, T3 = (a.ravel() for a in np.meshgrid(t, t, indexing="ij"))
    W = (w[:, None] * w[None, :]).ravel()
    p, rho2, rho4 = [], [], []
    for r in radii:
        rc = sl * r
        x2 = rc * T2
        x3 = x2 * T3
        x = np.stack([np.full_like(x2, rc), x2, x3])
        d = W * rc * x2 * slice_measure(*x) * dens(x)
        q = (x2**2 + x3**2) / dens.lam
        p.append(sl * K0 * d.sum())
        rho2.append((d * q).sum() / d.sum())
        rho4.append((d * q**2).sum() / d.sum())
    return np.array(p), np.array(rho2), np.array(rho4)


def grid(spec):
    """Samples from "first:last:step" (last included when on the grid) or a comma-separated list."""
    if ":" not in spec:
        return np.array([float(v) for v in spec.split(",")])
    first, last, step = map(float, spec.split(":"))
    if step <= 0 or last < first:
        raise ValueError("a grid first:last:step needs step > 0 and last >= first")
    return np.round(np.arange(first, last + step * 1e-8, step), 12)


def shard(spec=None):
    """(i, n) from "i/n", by default 0/1 (everything)."""
    spec = spec or "0/1"
    index, count = map(int, spec.split("/"))
    if not 0 <= index < count:
        raise ValueError("a shard i/n needs 0 <= i < n")
    return index, count


def local_power(r, p):
    """-d ln p / d ln r, from a cubic spline of ln p against ln r."""
    spline = CubicSpline(np.log(r), np.log(p))
    return lambda x: -spline(np.log(x), 1)


def density(args):
    start = time.time()
    state = load(args.state)
    dens = Density(state, OperatorLibrary(args.cache / OPERATORS))
    K0 = jacobian_constant()
    task, count = shard(args.shard)
    radii = grid(args.radii)[task::count]
    print(f"{len(state.basis.functions)} functions, {dens.K} cores, {dens.cols.shape[0]} core monomials, "
          f"lambda = {dens.lam:.9f}; {len(radii)} radii; setup {time.time() - start:.0f}s", flush=True)
    out = {"state": str(args.state), "lambda": dens.lam, "r": radii.tolist()}
    for order in map(int, args.orders.split(",")):
        p, rho2, _ = shape_integrals(dens, radii, order, K0)
        out[f"p{order}"], out[f"rho2_{order}"] = p.tolist(), rho2.tolist()
        print(f"order {order} done ({time.time() - start:.0f}s)", flush=True)
    Path(args.out.format(task=task)).write_text(json.dumps(out))


def norm(args):
    """Norm and normalized size moments, by Gauss in r over (0, rmax).

    r is the largest singular value, rho^2 is the sum of the other two squared singular values,
    and R^2 = r^2 + rho^2 is the total squared radius reported by bfss measure.
    """
    start = time.time()
    state = load(args.state)
    dens = Density(state, OperatorLibrary(args.cache / OPERATORS))
    t, w = np.polynomial.legendre.leggauss(args.nodes)
    r, w = (t + 1) * args.rmax / 2, w * args.rmax / 2
    p, rho2, _ = shape_integrals(dens, r, args.order, jacobian_constant())
    total = float(np.sum(w * p))
    r2 = float(np.sum(w * p * r**2) / total)
    out = {"state": str(args.state), "norm": total, "r2": r2, "r_rms": float(np.sqrt(r2)),
           "rho2": float(np.sum(w * p * rho2) / total),
           "R2": float(np.sum(w * p * (r**2 + rho2)) / total),
           "order": args.order, "nodes": args.nodes, "rmax": args.rmax}
    Path(args.out).write_text(json.dumps(out))
    print(f"int p dr = {total:.10f}, <r^2> = {r2:.8f}, rms r = {out['r_rms']:.8f}, "
          f"<rho^2> = {out['rho2']:.8f}, <R^2> = {out['R2']:.8f} ({time.time() - start:.0f}s)")


def window_fit(spline, r0, r1, n=81):
    """Fits of r^10 p on [r0, r1]; spline is ln(r^10 p) against ln r."""
    z = np.linspace(np.log(r0), np.log(r1), n)
    lnq = spline(z)
    c_fit = np.exp(lnq.mean())
    return {"r0": float(r0), "r1": float(r1), "C_fit_over_C": float(c_fit / C_PRED),
            "scatter": float(np.sqrt(np.mean((np.exp(lnq) / c_fit - 1) ** 2))),
            "free_power": float(10 - np.polyfit(z, lnq, 1)[0]), "spread": float(np.std(lnq))}


def fit(args):
    shards = [json.loads(Path(f).read_text()) for f in args.shards]
    r = np.concatenate([s["r"] for s in shards])
    order = np.argsort(r)
    r = r[order]
    have = [{int(k[1:]) for k in s if k.startswith("p") and k[1:].isdigit()} for s in shards]
    orders = sorted(set.intersection(*have))  # orders computed at every radius; the highest is used
    p = {o: np.concatenate([s[f"p{o}"] for s in shards])[order] for o in orders}
    rho2 = np.concatenate([s[f"rho2_{orders[-1]}"] for s in shards])[order]
    best = p[orders[-1]]
    out = {"orders": orders, "r": r.tolist(), "p": best.tolist(), "rho2": rho2.tolist(),
           "order_agreement": {f"{o}/{orders[-1]}": float(np.max(np.abs(p[o] / best - 1))) for o in orders[:-1]}}
    for o in sorted(set.union(*have) - set(orders)):  # higher orders computed on part of the grid
        part = [s for s in shards if f"p{o}" in s]
        a = np.concatenate([s[f"p{o}"] for s in part])
        b = np.concatenate([s[f"p{orders[-1]}"] for s in part])
        rr = np.concatenate([s["r"] for s in part])
        out["order_agreement"][f"{orders[-1]}/{o}"] = {"radii": [float(rr.min()), float(rr.max())],
                                                       "max_rel": float(np.max(np.abs(b / a - 1)))}
    if args.norm:
        out["norm_check"] = json.loads(Path(args.norm).read_text())
    spline = CubicSpline(np.log(r), np.log(r**10 * best))
    power = local_power(r, best)
    fine = np.round(np.arange(r[0], r[-1] + 1e-9, 0.05), 3)
    out["local_power"] = [[float(x), float(power(x))] for x in fine]
    out["r10p_over_C"] = [[float(x), float(np.exp(spline(np.log(x))) / C_PRED)] for x in fine]
    if args.compare:
        lo, hi = map(float, args.compare.split(","))
        local = power(np.linspace(lo, hi, 2001))
        out["compare"] = {"r0": lo, "r1": hi, "local_power_min": float(local.min()), "local_power_max": float(local.max())}
    if args.window:
        out["fixed_window"] = window_fit(spline, *map(float, args.window.split(",")))
    out["windows"] = {}
    for ratio in RATIOS:
        scan = [window_fit(spline, r0, r0 * ratio) for r0 in np.round(np.arange(r[0], r[-1] / ratio + 1e-9, 0.01), 2)]
        out["windows"][str(ratio)] = {"flattest": min(scan, key=lambda f: f["spread"]), "scan": scan}
    Path(args.out).write_text(json.dumps(out, indent=1))
    f = out["windows"]["1.1"]["flattest"]
    print(f"orders {orders}, agreement {out['order_agreement']}")
    print(f"flattest window of ratio 1.1: {f['r0']:.2f}-{f['r1']:.3f}, C_fit/C = {f['C_fit_over_C']:.6f}, "
          f"scatter {100 * f['scatter']:.2f}%, free power {f['free_power']:.4f}")
    if args.compare:
        c = out["compare"]
        print(f"local power on {c['r0']}-{c['r1']}: {c['local_power_min']:.4f}-{c['local_power_max']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="directory of the operator library")
    sub = parser.add_subparsers(dest="command", required=True)
    den = sub.add_parser("density", help="p(r) and <rho^2>_r on a grid of physical radii")
    den.add_argument("state", type=Path)
    den.add_argument("out", help="output JSON; {task} is replaced by the shard index")
    den.add_argument("--radii", default="3:8:0.025", help="the physical radii, first:last:step or comma-separated")
    den.add_argument("--orders", default="32,48,64", help="orders of the shape quadrature")
    den.add_argument("--shard", help="i/n: every n-th radius from the i-th (default: 0/1)")
    nor = sub.add_parser("norm", help="norm and size moments <r^2>, rms r, <rho^2>, and <R^2>")
    nor.add_argument("state", type=Path)
    nor.add_argument("out")
    nor.add_argument("--order", type=int, default=48)
    nor.add_argument("--nodes", type=int, default=160)
    nor.add_argument("--rmax", type=float, default=10.0)
    fi = sub.add_parser("fit", help="fits of r^10 p from density shards")
    fi.add_argument("out")
    fi.add_argument("shards", nargs="+")
    fi.add_argument("--norm", help="output of the norm command, recorded with the fits")
    fi.add_argument("--compare", help="r0,r1: report the range of the local power -d ln p / d ln r there")
    fi.add_argument("--window", help="r0,r1: also fit on this fixed window")
    args = parser.parse_args(argv)
    {"density": density, "norm": norm, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
