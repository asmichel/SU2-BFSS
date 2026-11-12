"""Assemble EFT density shards into a record, with numerical cross-checks recorded rather than accepted by tolerances.

    python analysis/assemble_eft_density.py --radial bulk-*.json --joint joint-*.json \
        --transverse moments-*.json --conditional conditional.json \
        --tail results/extended_tail_fit.json --out results/extended_eft_density.json

Shards of another state (by SHA-256) and incomplete shard sets are errors.  The radial section keeps its own
quadrature orders; the tail record of the same state supplies the finer, higher-order radial grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

from eft_density import agreement

METADATA = ("schema_version", "state", "state_sha256", "lambda", "n_functions",
            "jacobian_constant", "coordinates", "normalization", "joint_array_axes")


def read_record(path, reference=None):
    """A JSON record; with a reference record, it must be of the same state."""
    record = json.loads(Path(path).read_text())
    if reference is not None and record.get("state_sha256") != reference["state_sha256"]:
        raise ValueError(f"{path}: state SHA-256 mismatch")
    return record


def combine(paths, task, reference):
    """Join the radius shards of a task, sorted by radius; transverse files hold radial sections."""
    records = [read_record(path, reference) for path in paths]
    shards = [record.get("shard") for record in records]
    if any(shard is not None for shard in shards) and (
            any(shard is None or shard["count"] != len(paths) for shard in shards)
            or sorted(shard["index"] for shard in shards) != list(range(len(paths)))):
        raise ValueError(f"{task}: incomplete or duplicate shard indices")
    sections = [record.get(task, record.get("radial")) for record in records]
    orders = sections[0]["orders"]
    fields = sorted(sections[0]["by_order"][str(orders[-1])])
    r = np.concatenate([np.asarray(section["r"], dtype=float) for section in sections])
    permutation = np.argsort(r)
    by_order = {str(order): {key: np.concatenate([np.asarray(s["by_order"][str(order)][key]) for s in sections])
                             [permutation].tolist() for key in fields} for order in orders}
    result = {"r": r[permutation].tolist(), **by_order[str(orders[-1])], "orders": orders,
              "by_order": by_order, "order_agreement": agreement(by_order, orders)}
    if "rho" in sections[0]:
        result["rho"] = np.asarray(sections[0]["rho"], dtype=float).tolist()
    return result


def differences(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = actual - expected
    nonzero = np.abs(expected) > 0
    relative = difference[nonzero] / expected[nonzero]
    return {"max_abs": float(np.max(np.abs(difference))),
            "max_rel": float(np.max(np.abs(relative))) if len(relative) else None,
            "rms_rel": float(np.sqrt(np.mean(relative**2))) if len(relative) else None}


def matches(r, reference_r):
    pairs = [(i, j) for i, value in enumerate(r) for j, other in enumerate(reference_r)
             if abs(value - other) <= 1e-9]
    if not pairs:
        raise ValueError("no matching radii for an independent comparison")
    return np.asarray(pairs, dtype=int).T


def compare_tail(section, tail):
    i, j = matches(section["r"], tail["r"])
    return {"r": np.asarray(section["r"])[i].tolist(),
            **{key: differences(np.asarray(section[key])[i], np.asarray(tail[key])[j])
               for key in ("p", "rho2")}}


def conditional_checks(section, transverse):
    rho = np.asarray(section["rho"])
    density = np.asarray(section["p_cond"])
    result = {"integration_range": [float(rho[0]), float(rho[-1])]}
    for name, integrate in (("trapezoid", np.trapezoid), ("simpson", simpson)):
        mass = float(integrate(density, x=rho))
        metrics = {"mass": mass, "mass_minus_one": mass - 1}
        for power in (2, 4):
            moment = float(integrate(density * rho**power, x=rho))
            expected = section[f"rho{power}"]
            metrics[f"rho{power}"] = moment
            metrics[f"rho{power}_relative_error"] = moment / expected - 1
            metrics[f"normalized_rho{power}_relative_error"] = moment / mass / expected - 1 if mass else None
        result[name] = metrics
    i, _ = matches(transverse["r"], [section["r"]])
    result["against_transverse"] = {
        key: differences([section["p_radial"] if key == "p" else section[key]],
                         np.asarray(transverse[key])[i]) for key in ("p", "rho2", "rho4")}
    return result


def joint_checks(section, tail):
    rho = np.asarray(section["rho"])
    i, j = matches(section["r"], tail["r"])
    expected = np.asarray(tail["p"])[j]
    resolved = expected >= np.max(expected) * 1e-10
    i, j, expected = i[resolved], j[resolved], expected[resolved]
    density = np.asarray(section["p"])[i]
    result = {"r": np.asarray(section["r"])[i].tolist(),
              "integration_range": [float(rho[0]), float(rho[-1])],
              "resolved_density_relative_floor": 1e-10}
    for name, integrate in (("trapezoid", np.trapezoid), ("simpson", simpson)):
        marginal = integrate(density, x=rho, axis=1)
        raw_rho2 = integrate(density * rho**2, x=rho, axis=1)
        result[name] = {"p": differences(marginal, expected),
                        "rho2": differences(raw_rho2 / marginal, np.asarray(tail["rho2"])[j])}
    return result


def read_tail(path, reference):
    """The profile of a tail record of the reference's state, with its conditional <rho^2> as rho2 (a curated
    record names it rho2_mean, a radial_tail.py fit output rho2)."""
    record = read_record(path, reference)
    if "lambda" in record and not np.isclose(record["lambda"], reference["lambda"], rtol=1e-13, atol=0):
        raise ValueError(f"{path}: inconsistent lambda")
    tail = dict(record.get("profile", record))
    tail.setdefault("rho2", tail.get("rho2_mean"))
    return tail


def validation(result, tail):
    """The residuals of the assembled sections against each other and against the tail profile."""
    return {
        "status": "numerical residuals recorded for review; no tolerance acceptance inferred",
        "tail_dilation_identity": "matching saved-state SHA-256; explicit lambda checked when supplied",
        "radial_vs_tail": compare_tail(result["radial"], tail),
        "transverse_vs_tail": compare_tail(result["transverse"], tail),
        "conditional_integrals": conditional_checks(result["conditional"], result["transverse"]),
        "joint_integrals_vs_tail": joint_checks(result["joint"], tail),
    }


def assemble(radial_paths, joint_paths, transverse_paths, conditional_path, tail_path):
    reference = read_record(radial_paths[0])
    result = {key: reference[key] for key in METADATA}
    for name, paths in (("radial", radial_paths), ("joint", joint_paths), ("transverse", transverse_paths)):
        result[name] = combine(paths, name, reference)
    conditional_record = read_record(conditional_path, reference)
    if conditional_record.get("shard", {"index": 0, "count": 1}) != {"index": 0, "count": 1}:
        raise ValueError("conditional input must contain its full rho grid, not a shard")
    conditional = conditional_record["conditional"]
    conditional["order_agreement"] = agreement(conditional["by_order"], conditional["orders"])
    result["conditional"] = conditional
    result["validation"] = validation(result, read_tail(tail_path, reference))
    result["sources"] = {"radial": list(map(str, radial_paths)), "joint": list(map(str, joint_paths)),
                         "transverse": list(map(str, transverse_paths)), "conditional": str(conditional_path),
                         "tail": str(tail_path), "tail_sha256": hashlib.sha256(Path(tail_path).read_bytes()).hexdigest()}
    result["assembly"] = "analysis/assemble_eft_density.py"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("radial", "joint", "transverse"):
        parser.add_argument(f"--{name}", type=Path, nargs="+", required=True)
    for name in ("conditional", "tail", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble(args.radial, args.joint, args.transverse, args.conditional, args.tail)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
