"""Shared fixtures: published data, the operator library and a small exact contraction table."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from bfss.contract import OPERATORS, Core, OperatorLibrary
from bfss.covariants import SeedSet
from bfss.hilbert import Basis, Expansion, ProjectedGaussian, TrialFunction
from bfss.reference import companion_files, load_reference
from bfss.state import Engine

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
# The published companion files, copied verbatim from the paper's code addendum (tests/data/reference/README.md).
REFERENCE = ROOT / "tests/data/reference"


@pytest.fixture(scope="session")
def reference_source() -> Path:
    return REFERENCE


@pytest.fixture(scope="session")
def companion(reference_source) -> dict:
    """The parsed companion files: coefficients.json, fermion-data.json, moments.json."""
    return companion_files(reference_source)


@pytest.fixture(scope="session")
def reference(reference_source):
    return load_reference(reference_source)


@pytest.fixture(scope="session")
def seeds(reference) -> SeedSet:
    return reference[0]


@pytest.fixture(scope="session")
def library() -> OperatorLibrary:
    return OperatorLibrary(CACHE / OPERATORS)


@pytest.fixture(scope="session")
def small_seeds(seeds) -> SeedSet:
    """S, U0 and O: tensor, expression and slice-formula (kind "canonical") definitions."""
    keep = [seeds.by_name[n] for n in ("S", "U0", "O")]
    return SeedSet(keep, seeds.fields, seeds.norm)


SMALL_CORES = [Core(0, 0), Core(0, 1), Core(0, 2), Core(0, 3), Core(1, 0), Core(1, 1), Core(2, 0)]


@pytest.fixture(scope="session")
def engine(library) -> Engine:
    return Engine(CACHE)


@pytest.fixture(scope="session")
def small_basis(small_seeds) -> Basis:
    """Canonical-width basis whose cores are SMALL_CORES (widths 2, 3; prefactors 1, s)."""
    return Basis(small_seeds, [TrialFunction(ProjectedGaussian(a, 1), e, c.power, small_seeds.seeds[c.seed].name)
                               for a in (2, 3) for e in ((0, 0, 0), (1, 0, 0)) for c in SMALL_CORES])


def random_expansion(basis, seed):
    """An expansion over a basis with random coefficients, multiples of 1/997."""
    values = np.random.default_rng(seed).integers(-500, 500, len(basis))
    return Expansion(basis, [Decimal(int(v)) / 997 for v in values])


@pytest.fixture(scope="session")
def two_width_state(engine, small_seeds):
    """A state over two longitudinal widths with prefactors 1, s, dilated by 3/2, and its measurement by the
    256-bit ball engine."""
    widths = ((Fraction(2), Fraction(1)), (Fraction(3, 2), Fraction(1, 2)))
    basis = Basis(small_seeds, [TrialFunction(ProjectedGaussian(a, b), e, c.power, small_seeds.seeds[c.seed].name)
                                for a, b in widths for e in ((0, 0, 0), (1, 0, 0)) for c in SMALL_CORES])
    state = random_expansion(basis, 11).dilated(Fraction(3, 2))
    return state, engine.measure(state)


@pytest.fixture(scope="session")
def small_table(engine, small_basis):
    table = engine.table(small_basis)
    assert [(c.seed, c.power) for c in table.cores] == [(c.seed, c.power) for c in SMALL_CORES]
    return table


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
                     help="also run the exhaustive checks: brute-force Fock-space and quadrature oracles, the long "
                          "algebraic identities, and end-to-end checks against the published 1951-function state")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="exhaustive check: run with --slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
