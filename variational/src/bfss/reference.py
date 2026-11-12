"""Import of the published companion data (coefficients.json, fermion-data.json, moments.json).

The three files are read from a directory holding them.  The reference format writes seeds as
definition strings over the same vocabulary as ``covariants`` except for P_hat, which it states in
index notation; its index-free form in terms of the operator-valued colour matrix
YM(X)_dc = sum_i X_di K_ci is

    P_hat = -2 tr(YM^T YM) + 160 s - 2 Y^2 + 2 tr(YM YM).
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from .covariants import ExpressionSeed, SeedSet, TensorSeed

P_HAT = "-2*tr(YM(X)^T*YM(X))+160*s*I-2*Y(X)^2+2*tr(YM(X)*YM(X))"


def seeds_from_reference(fermion: dict) -> SeedSet:
    tensors = {
        "singlet": [((0, 0, 0), fermion["singlet"]["tensor"])],
        "quadratic_seed": list(zip(map(tuple, fermion["quadratic_seed"]["powers"]), fermion["quadratic_seed"]["tensors"])),
    }
    denominators = {"singlet": int(fermion["singlet"]["tensor_denominator"]),
                    "quadratic_seed": int(fermion["quadratic_seed"]["tensor_denominator"])}
    seeds = []
    for s in fermion["seeds"]:
        name, degree, definition = s["name"], s["cartesian_degree"], s["definition"]
        source = "quadratic_seed" if "quadratic_seed" in definition else "singlet" if "singlet" in definition else None
        if source:
            seeds.append(TensorSeed(name, degree, tensors[source], denominators[source]))
        elif definition.startswith("At D(x,y,z):"):
            formula = re.match(r"At D\(x,y,z\):\s*(.*?);", definition).group(1)
            seeds.append(ExpressionSeed(name, degree, formula, slice_formula=True))
        else:
            seeds.append(ExpressionSeed(name, degree, definition.replace("P_hat(X)", "P_hat")))
    return SeedSet(seeds, {"P_hat": P_HAT}, norm=Fraction(1, 1144))


COMPANION_FILES = ("coefficients.json", "fermion-data.json", "moments.json")


def companion_files(directory: Path) -> dict:
    """The parsed companion files of a directory holding them."""
    return {name: json.loads((Path(directory) / name).read_text()) for name in COMPANION_FILES}


def load_reference(source: Path):
    """(seed set, parsed coefficients.json, moments) from the published data."""
    files = companion_files(source)
    return seeds_from_reference(files["fermion-data.json"]), files["coefficients.json"], files["moments.json"]


def reference_state(source: Path):
    """The published state as an ``Expansion`` over its physical (dilated) trial wavefunctions."""
    from decimal import Decimal

    from .hilbert import Basis, Expansion, ProjectedGaussian, TrialFunction

    seeds, coeffs, moments = load_reference(source)
    functions = []
    for m in coeffs["modes"]:
        c = m["covariant"]
        functions.append(TrialFunction(ProjectedGaussian(Fraction(m["a"]), Fraction(m["b"])),
                                       (c["radial"], c["shape"], c["volume"]), c["power"],
                                       coeffs["seed_names"][c["seed"]]))
    meta = {"dilation": coeffs["dilation"], "source": "published companion data"}
    return Expansion(Basis(seeds, functions), [Decimal(m["coefficient"]) for m in coeffs["modes"]], meta)
