"""Curate the outputs of radial_tail.py into a tail record shaped like results/seeded_tail_fit.json.

    python analysis/curate_tail_fit.py FIT.json NORM.json STATE.json MEASUREMENT.json CODE_COMMIT OUT.json [--check CHECK.json]

FIT and NORM are the outputs of radial_tail.py fit and norm, MEASUREMENT is the state's measurement record and
CODE_COMMIT the commit that evaluated the density.  CHECK is a radial_tail.py density shard on part of the grid at
the profile's order and a higher one; it gives quadrature_check_<order>_<higher>.  The record's generation field
(commands) is not derived from these files and is added by hand.  results/extended_tail_fit.json
predates this script and has another shape (method descriptions, numeric_encoding and a validation section; no
lambda, generation or quadrature_check).
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from radial_tail import local_power


def quadrature_check(check, fit):
    """Agreement of the profile's order with a higher order on the radii of CHECK, which must reproduce the profile."""
    low, high = sorted(int(k[1:]) for k in check if k.startswith("p") and k[1:].isdigit())
    if low != fit["orders"][-1]:
        raise ValueError(f"the check's lower order {low} is not the profile's order {fit['orders'][-1]}")
    r, p_low, p_high = (np.array(check[k]) for k in ("r", f"p{low}", f"p{high}"))
    profile = dict(zip(np.round(fit["r"], 9), fit["p"]))
    deviation = max(abs(profile[x] / p - 1) for x, p in zip(np.round(r, 9), p_low))
    if deviation > 1e-10:
        raise ValueError(f"the check's order-{low} values differ from the profile's by {deviation:.1e}")
    step = np.round(r[1] - r[0], 9)
    return f"quadrature_check_{low}_{high}", {
        "radii": r.tolist(), "max_rel": float(np.max(np.abs(p_low / p_high - 1))),
        "note": f"separate run at orders {low} and {high} on {r[0]:g}:{r[-1]:g}:{step:g}; "
                f"its order-{low} values agree with the profile's to {deviation:.0e}"}


def local_power_band(r, p):
    """(r, value) of the minimum of the local power -d ln p/d ln r away from the bulk, and the radii on either side
    where it is 10.2 (None where it stays below)."""
    spline_power = local_power(r, p)
    power = lambda x: float(spline_power(x))
    # the local power falls from ~13 at r = 3 to a minimum near 10
    coarse = np.linspace(4.0, r[-1] - 0.5, 2001)
    x0 = coarse[np.argmin([power(x) for x in coarse])]
    m = minimize_scalar(power, bounds=(x0 - 0.2, x0 + 0.2), method="bounded", options={"xatol": 1e-10})
    rmin, pmin = float(m.x), float(m.fun)
    lo = brentq(lambda x: power(x) - 10.2, 3.5, rmin) if power(3.5) > 10.2 else None
    hi = brentq(lambda x: power(x) - 10.2, rmin, r[-1]) if power(r[-1]) > 10.2 else None
    return rmin, pmin, lo, hi


def record(fit, norm, state_path, meas, commit, check=None):
    state_sha = hashlib.sha256(Path(state_path).read_bytes()).hexdigest()
    if meas["state_sha256"] != state_sha:
        raise ValueError("the measurement is of another state")
    r, p = np.array(fit["r"]), np.array(fit["p"])
    rmin, pmin, lo, hi = local_power_band(r, p)
    windows = fit["windows"]
    keys = ("r0", "r1", "C_fit_over_C", "free_power", "scatter")
    flat = {ratio: {k: v["flattest"][k] for k in keys} for ratio, v in windows.items()}
    inside = [w for w in windows["1.1"]["scan"]
              if lo is not None and hi is not None and w["r0"] >= lo - 1e-9 and w["r1"] <= hi + 1e-9]
    span = lambda items, k: [min(w[k] for w in items), max(w[k] for w in items)]
    out = {
        "code_commit": commit,
        "analysis": "analysis/radial_tail.py (density, norm and fit commands)",
        "state_file": str(state_path),
        "state_sha256": state_sha,
        "lambda": float(Fraction(json.loads(Path(state_path).read_text())["metadata"]["dilation"])),
        "observable": "p(r) = int d^27X |psi(X)|^2 delta(r - sigma_1(X)), the density of the largest singular value r",
        "benchmark": {"density": "C r^-10", "C": "1890 sqrt(2)"},
        "method": {"radii": [float(r[0]), float(r[-1])], "grid_step": float(np.round(r[1] - r[0], 6)),
                   "quadrature_orders": fit["orders"], "order_used": fit["orders"][-1],
                   "higher_order_check": {k: v for k, v in fit["order_agreement"].items() if isinstance(v, dict)}},
        "quadrature_agreement": fit["order_agreement"],
        "normalization_check": {**{k: norm[k] for k in ("norm", "R2", "order", "nodes", "rmax")},
                                "R2_measured_256bit": meas["observables"]["R2"]},
        "reported": {
            **{k: flat["1.1"][k] for k in keys},
            "local_power_minimum": {"value": pmin, "r": rmin},
            "local_power_within_2_percent_of_10": [lo, hi],
            "flattest_windows_ratio_1.05_to_1.5": {k: span(flat.values(), k) for k in ("C_fit_over_C", "free_power")},
            "ratio_1.1_windows_inside_2_percent_band": {
                "count": len(inside), **{k: span(inside, k) if inside else None for k in ("C_fit_over_C", "free_power")}},
        },
        "derived": [
            "reported window and fits: windows['1.1'].flattest of the fit output",
            "local_power_minimum: bounded minimization of -d ln p/d ln r, from CubicSpline(ln r, ln p) of the fit output's r and p",
            "local_power_within_2_percent_of_10: the roots of that local power = 10.2 on either side of its minimum",
            "flattest_windows_ratio_1.05_to_1.5: range over windows[ratio].flattest of the six ratios",
            "ratio_1.1_windows_inside_2_percent_band: range over the windows['1.1'].scan entries inside local_power_within_2_percent_of_10",
            "window_variations.sliding_ratio_1.1: every fifth window of windows['1.1'].scan (r0 step 0.05); spread omitted"],
        "window_variations": {"flattest_by_ratio": flat,
                              "sliding_ratio_1.1": [{k: w[k] for k in keys} for w in windows["1.1"]["scan"][::5]]},
        "profile": {"r": fit["r"], "p": fit["p"], "rho2_mean": fit["rho2"],
                    "local_power": fit["local_power"], "r10p_over_C": fit["r10p_over_C"]},
    }
    if check is not None:
        key, value = quadrature_check(check, fit)
        out[key] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("fit", "norm", "state", "measurement"):
        parser.add_argument(name, type=Path)
    parser.add_argument("commit")
    parser.add_argument("out", type=Path)
    parser.add_argument("--check", type=Path, help="radial_tail.py density shard at the profile's order and a higher one")
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text())
    out = record(read(args.fit), read(args.norm), args.state, read(args.measurement), args.commit,
                 read(args.check) if args.check else None)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["reported"], indent=1))
    print("quadrature", out["quadrature_agreement"])
    print("norm", out["normalization_check"])


if __name__ == "__main__":
    main()
