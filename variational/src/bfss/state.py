"""Persistence and measurement of variational states.

A state file contains everything needed to rebuild the wavefunction with the
generic machinery: the seed definitions (tensor data and expressions), the ordered
trial wavefunctions (profile widths, invariant exponents (k, l, m), Yukawa power, seed), the real
coefficients multiplying exactly those functions, and descriptive metadata.

Measurement uses the reduced blocks at the canonical widths (largest longitudinal width
b = 1) and the scaling of each block under the dilation U_lambda.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction
from math import ceil, log10
from pathlib import Path

import flint
from flint import arb

from .blocks import BLOCKS, OBSERVABLES
from .contract import OPERATORS, Core, contract
from .covariants import SeedSet
from .hilbert import Basis, Expansion, TrialFunction
from .model import NU
from .scalars import RATIONALS, Reals, ball
from .variational import KAPPA, ReducedBlocks, optimize

SCHEMA = "bfss.state.v1"


def write_json(path: Path, data, width=200):
    """Indented JSON in which small containers (a tensor row, a trial wavefunction) stay on one line."""

    def fmt(value, indent):
        flat = json.dumps(value)
        if not isinstance(value, (dict, list)) or len(flat) + indent <= width:
            return flat
        pad = " " * (indent + 1)
        items = ([f"{pad}{json.dumps(k)}: {fmt(v, indent + 1)}" for k, v in value.items()] if isinstance(value, dict)
                 else [pad + fmt(v, indent + 1) for v in value])
        brackets = "{}" if isinstance(value, dict) else "[]"
        return brackets[0] + "\n" + ",\n".join(items) + "\n" + " " * indent + brackets[1]

    Path(path).write_text(fmt(data, 0) + "\n")


def save(state: Expansion, path: Path):
    write_json(path, {
        "schema": SCHEMA,
        "wavefunction": "sum_j coefficients[j] * f_(a_j,b_j)(X) * s^k u^l v^m * Y(X)^power * seed(X), (k, l, m) = invariant",
        "seeds": state.basis.seeds.to_json(),
        "basis": [f.to_json() for f in state.basis.functions],
        "coefficients": [str(c) for c in state.coefficients],
        "metadata": state.metadata,
    })


def load(path: Path) -> Expansion:
    data = json.loads(Path(path).read_text())
    assert data["schema"] == SCHEMA, f"unknown schema {data['schema']}"
    basis = Basis(SeedSet.from_json(data["seeds"]), [TrialFunction.from_json(f) for f in data["basis"]])
    return Expansion(basis, [Decimal(c) for c in data["coefficients"]], data.get("metadata", {}))


def quarter_root(b):
    """b^{1/4}: reduced trial wavefunctions are tilde psi_j = b_j^{1/4} pi^{-nu/2} f_j P_j (see blocks)."""
    return ball(b).sqrt().sqrt()


class Engine:
    """Caches the operator library and contraction tables, and evaluates/optimizes states.

    Matrix elements are evaluated as balls of ``prec`` bits, or exactly with ``exact=True``;
    the orthonormalization and the observables use ``prec`` bits in either case.  Dense
    products run on ``threads`` FLINT threads (default: all CPUs).
    """

    def __init__(self, cache_dir: Path, workers=None, prec=256, exact=False, threads=None):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers, self.prec, self.threads = workers, prec, threads or os.cpu_count() or 1
        self.digits = ceil(prec * log10(2)) + 8  # decimal digits of stored and dilated coefficients
        self.scalars = RATIONALS if exact else Reals(prec)
        self._blocks = None  # (basis key, ReducedBlocks) of the last canonical basis

    @contextmanager
    def arithmetic(self):
        """Working precision and FLINT threads for the number crunching."""
        saved, flint.ctx.threads = flint.ctx.threads, self.threads
        try:
            with flint.ctx.workprec(self.prec):
                yield
        finally:
            flint.ctx.threads = saved

    def table(self, basis: Basis):
        """Contraction table for the cores of a basis, computed once and cached.

        Any cached table of the same seeds whose cores include them serves as well: its exact
        coefficients do not depend on the other cores.  An index file next to each table
        records its seeds and cores.
        """
        digest = lambda data: hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        cores, seeds = [[s, p] for s, p in basis.cores()], digest(basis.seeds.to_json())
        indices = {f.with_suffix(".pkl"): json.loads(f.read_text()) for f in self.cache.glob("contractions-*.json")}
        found = [path for path, i in indices.items()
                 if i["seeds"] == seeds and set(map(tuple, cores)) <= set(map(tuple, i["cores"]))]
        if found:
            return pickle.loads(min(found, key=lambda path: len(indices[path]["cores"])).read_bytes())
        path = self.cache / f"contractions-{digest([basis.seeds.to_json(), cores])}.pkl"
        table = contract(basis.seeds, [Core(s, p) for s, p in cores], self.cache / OPERATORS, self.workers)
        path.write_bytes(pickle.dumps(table))
        path.with_suffix(".json").write_text(json.dumps({"seeds": seeds, "cores": [[c.seed, c.power] for c in table.cores]}))
        return table

    def blocks(self, basis: Basis) -> ReducedBlocks:
        """Reduced blocks of a canonical basis; those of the last basis are kept (optimize, then measure)."""
        assert basis.reference_width() == 1, "reduced blocks are taken at the canonical widths (largest b = 1)"
        key = (id(basis.seeds), tuple(basis.functions))
        if self._blocks is None or self._blocks[0] != key:
            self._blocks = key, ReducedBlocks(basis, self.table(basis), self.scalars)
        return self._blocks[1]

    def measure(self, state: Expansion) -> dict:
        """M (squared norm), E = <H>, Haux = <H_aux>, F_lb (a float: the bound [1 - kappa <H_aux>]_+ on the
        overlap F), T, u, Y (<T>, <u>, <Y>, the three terms of H), R2 to R8 (<R^2> to <R^8>) and canonical_blocks
        of a (dilated) state.

        U_{1/lambda} takes the state to the canonical widths (largest b = 1), where an observable
        is sum_B lambda^{e_B} <B> over the reduced blocks.
        """
        with self.arithmetic():
            lam = state.basis.reference_width()
            canonical = state.dilated(1 / lam, self.digits)
            blocks = self.blocks(canonical.basis)
            x = [arb(str(c)) / quarter_root(f.profile.b)
                 for c, f in zip(canonical.coefficients, canonical.basis.functions)]
            values = {k: blocks.quadratic_form(k, x) for k in BLOCKS}
            m = values["M"]
            canonical_blocks = {k: v / m for k, v in values.items()}
            physical = {name: obs.dilated(canonical_blocks, ball(lam))
                        for name, obs in OBSERVABLES.items() if name != "M"}
            haux = physical["Haux"]
            radial = {k: v for k, v in physical.items() if k[0] == "R"}  # <R^2> .. <R^8>
            return {"M": m * arb.pi() ** ball(NU), "E": physical["H"], "Haux": haux,
                    "F_lb": max(0.0, 1 - KAPPA * float(haux.mid())), **{k: physical[k] for k in ("T", "u", "Y")},
                    **radial, "canonical_blocks": canonical_blocks}

    def optimize(self, basis: Basis, dilation=None, **kwargs) -> Expansion:
        """Variational state minimizing <H_aux> (the paper's Eq. (20)) over the span and the dilation, or over the
        span at a given rational ``dilation``."""
        with self.arithmetic():
            result = optimize(self.blocks(basis), dilation=dilation, **kwargs)
            # a decimal fraction of the optimal dilation, or the given one
            lam = Fraction(f"{result.dilation:.16g}") if dilation is None else Fraction(dilation)
            scale = arb.pi() ** (-ball(NU) / 2)  # the reduced blocks omit pi^nu
            canonical = [Decimal((x * scale * quarter_root(f.profile.b)).mid().str(self.digits, radius=False))
                         for x, f in zip(result.x, basis.functions)]
        meta = {"dilation": str(lam), "canonical_width": "1", "Haux_eigenvalue": result.eigenvalue,
                "fidelity_lower_bound": result.fidelity_bound, "matrix_elements": repr(self.scalars)}
        return Expansion(basis, canonical, meta).dilated(lam, self.digits)
