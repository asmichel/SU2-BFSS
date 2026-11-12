"""Export radial and transverse densities of a saved state in physical coordinates.

    python analysis/eft_density.py radial STATE OUT --radii .1:3:.1 --orders 24,32
    python analysis/eft_density.py joint STATE OUT --radii .2:8:.2 --rho 0:3:.05 --orders 24,32
    python analysis/eft_density.py conditional STATE OUT --radius 5.6 --rho 0:3:.025 --orders 32,48

The radial task also exports the transverse moments against their leading EFT values (the paper, Sec. IV A):
rho2_over_eft = <rho^2>_r / (8/omega), omega = sqrt(2) r, and rho4_over_rho2_sq = <rho^4>_r / <rho^2>_r^2, whose EFT
value is 9/8.  --shard i/n splits the radii (radial, joint) or rho (conditional).  Outputs keep every quadrature order and identify the state by its SHA-256.  The density, the slice
measure, the quadrature and the grids are those of radial_tail.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bfss.contract import OPERATORS, OperatorLibrary
from bfss.state import load
from radial_tail import Density, grid, jacobian_constant, shape_integrals, shard, slice_measure, unit_rule


def shape_moments(dens, radii, order, K0):
    """p(r), the conditional physical moments <rho^2>_r, <rho^4>_r and their ratios to the EFT at physical radii."""
    radii = np.asarray(radii, dtype=float)
    p, rho2, rho4 = shape_integrals(dens, radii, order, K0)
    return {"p": p, "rho2": rho2, "rho4": rho4, "rho2_over_eft": radii * rho2 / (4 * np.sqrt(2)),
            "rho4_over_rho2_sq": rho4 / rho2**2}


def joint_density(dens, radii, rho, order, K0):
    """p(r,rho), shape (len(radii), len(rho)), including the polar Jacobian.

    x2=rho cos(theta), x3=rho sin(theta). The ordered chamber has theta<=pi/4
    and theta>=acos(r/rho) if rho>r, otherwise theta>=0. It is empty for
    rho>=sqrt(2)*r. In canonical coordinates the two density deltas supply
    lambda and dx2 dx3 supplies rhoc d(rhoc) dtheta.
    """
    radii, rho = np.asarray(radii, dtype=float), np.asarray(rho, dtype=float)
    t, w = unit_rule(order)
    sl = np.sqrt(dens.lam)
    out = np.zeros((len(radii), len(rho)))
    for i, r in enumerate(radii):
        valid = (rho > 0) & (rho < np.sqrt(2) * r)
        if not np.any(valid):
            continue
        rc, rhoc = sl * r, sl * rho[valid]
        lower = np.arccos(np.minimum(1, r / rho[valid]))
        length = np.pi / 4 - lower
        theta = lower[:, None] + length[:, None] * t
        x2, x3 = rhoc[:, None] * np.cos(theta), rhoc[:, None] * np.sin(theta)
        x = np.stack([np.full_like(x2, rc), x2, x3]).reshape(3, -1)
        values = (slice_measure(*x) * dens(x)).reshape(len(rhoc), order)
        out[i, valid] = dens.lam * K0 * rhoc * length * (values @ w)
    return out


def agreement(by_order, orders):
    """Report all-point relative error and error on values >=1e-10 of the peak."""
    best = by_order[str(orders[-1])]
    result = {}
    for order in orders[:-1]:
        fields = {}
        for key, high in best.items():
            high = np.asarray(high, dtype=float)
            diff = np.abs(np.asarray(by_order[str(order)][key]) - high)
            nonzero = np.abs(high) > 0
            resolved = np.abs(high) >= np.max(np.abs(high)) * 1e-10
            resolved &= nonzero
            fields[key] = {
                "max_abs": float(np.max(diff)),
                "max_rel": float(np.max(diff[nonzero] / np.abs(high[nonzero]))) if np.any(nonzero) else None,
                "max_rel_resolved": (float(np.max(diff[resolved] / np.abs(high[resolved])))
                                     if np.any(resolved) else None),
            }
        result[f"{order}/{orders[-1]}"] = fields
    return result


def evaluate(dens, K0, task, radii, rho=None, orders=(24, 32)):
    """One JSON-ready section of a task, with every quadrature order for the convergence checks.

    A radial section on the comparison window is stored as 'transverse'.  A conditional task has one radius; its p
    is the joint density and p_cond = p/p_radial a density with respect to physical rho.
    """
    radii, orders = np.asarray(radii, dtype=float), sorted(set(int(order) for order in orders))
    by_order = {}
    for order in orders:
        if task == "radial":
            values = shape_moments(dens, radii, order, K0)
        elif task == "joint":
            values = {"p": joint_density(dens, radii, rho, order, K0)}
        else:
            moments = shape_moments(dens, radii, order, K0)
            p = joint_density(dens, radii, rho, order, K0)[0]
            values = {key: value[0] for key, value in moments.items() if key != "p"}
            values.update(p=p, p_cond=p / moments["p"][0], p_radial=moments["p"][0])
        by_order[str(order)] = {key: np.asarray(value).tolist() for key, value in values.items()}
        print(f"{task}: order {order} complete", flush=True)
    section = {"r": float(radii[0]) if task == "conditional" else radii.tolist(),
               **by_order[str(orders[-1])], "orders": orders, "by_order": by_order,
               "order_agreement": agreement(by_order, orders)}
    if task != "radial":
        section["rho"] = np.asarray(rho, dtype=float).tolist()
    return section


def provenance(state_path, dens, K0):
    """Metadata shared by sections evaluated from one unchanged saved state."""
    state_path = Path(state_path)
    with state_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"schema_version": 1, "state": str(state_path), "state_sha256": digest,
            "lambda": dens.lam, "n_functions": len(dens.coef.c), "jacobian_constant": K0,
            "coordinates": "physical r=sigma_1, rho=sqrt(sigma_2^2+sigma_3^2)",
            "normalization": "raw saved-state density; conditional divided by its radial marginal",
            "joint_array_axes": ["r", "rho"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("task", choices=("radial", "joint", "conditional"))
    parser.add_argument("state", type=Path)
    parser.add_argument("out", help="output JSON; {task} expands to the shard index")
    parser.add_argument("--radii", help="radial/joint samples: first:last:step or comma-separated")
    parser.add_argument("--radius", type=float, default=5.6, help="conditional physical radius")
    parser.add_argument("--rho", help="joint/conditional samples: first:last:step or comma-separated")
    parser.add_argument("--orders", default="24,32")
    parser.add_argument("--shard", help="i/n; default 0/1")
    args = parser.parse_args(argv)
    if args.task != "conditional" and args.radii is None:
        parser.error("radial/joint tasks require an explicit --radii grid")
    if args.task != "radial" and args.rho is None:
        parser.error("joint/conditional tasks require an explicit --rho grid")
    radii = np.array([args.radius]) if args.task == "conditional" else grid(args.radii)
    rho = grid(args.rho) if args.rho else None
    index, count = shard(args.shard)
    if args.task == "conditional":
        rho = rho[index::count]
    else:
        radii = radii[index::count]
    start = time.time()
    state = load(args.state)
    dens = Density(state, OperatorLibrary(args.cache / OPERATORS))
    K0 = jacobian_constant()
    print(f"{len(dens.coef.c)} functions, lambda={dens.lam:.9f}; setup {time.time()-start:.1f}s", flush=True)
    record = provenance(args.state, dens, K0)
    record[args.task] = evaluate(dens, K0, args.task, radii, rho, [int(value) for value in args.orders.split(",")])
    record["shard"] = {"index": index, "count": count}
    out = Path(args.out.format(task=index))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {out} in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
