"""End-to-end checks against the published 1951-function state (run with ``pytest --slow``)."""

import pytest

from bfss.reference import reference_state
from bfss.state import Engine
from conftest import CACHE

pytestmark = pytest.mark.slow
PUBLISHED_NAMES = {"W": "u", "DK": "Haux_T", "DV": "Haux_u", "DY": "Haux_Y"}  # moments.json's names of the blocks


@pytest.fixture(scope="module")
def published(reference_source):
    return reference_state(reference_source)


@pytest.fixture(scope="module")
def canonical(published):
    return published.basis.dilated(1 / published.basis.reference_width())


@pytest.fixture(scope="module")
def primary(companion):
    return companion["moments.json"]["primary"]


def test_published_moments_are_reproduced(engine, published, primary):
    """Evaluation in 256-bit ball arithmetic agrees with the published primary evaluation to its own spread (~1e-6).

    The published primary and alternative evaluations differ from each other by up to
    1.1e-6 in these normalized blocks; E and <H_aux> cancel by five orders of magnitude, so
    they are compared through the blocks rather than directly.
    """
    measured = engine.measure(published)
    for name, value in primary["canonical_blocks"].items():
        block = measured["canonical_blocks"][PUBLISHED_NAMES.get(name, name)]
        assert float(block.mid()) == pytest.approx(float(value), rel=3e-6), name
    assert float(measured["M"].mid()) == pytest.approx(1, abs=1e-6)
    assert float(measured["R2"].mid()) == pytest.approx(float(primary["R2"]), rel=3e-6)
    assert float(measured["R4"].mid()) == pytest.approx(float(primary["R4"]), rel=3e-6)


def test_published_basis_has_no_exact_dependencies(canonical):
    """The published omissions are numerical: the exact Gram matrix has full rank."""
    exact = Engine(CACHE, exact=True)
    assert exact.blocks(canonical).independent_subset() == list(range(len(canonical)))


def test_optimization_from_scratch_on_the_published_span(engine, canonical, published):
    """Optimizing the published 1913-function span recovers the published fidelity bound.

    The published bound 0.990425 rests on matrices accurate to ~1e-6; the exact optimum on
    the same span lies within the reported evaluation sensitivity (4.4e-4) and improves on
    the published coefficients evaluated exactly.
    """
    omitted = [j for j, c in enumerate(published.coefficients) if c == 0]
    state = engine.optimize(canonical, omit=omitted)
    bound = state.metadata["fidelity_lower_bound"]
    assert abs(bound - 0.990425088640270657) < 4.4e-4
    assert bound > engine.measure(published)["F_lb"]
    assert all(state.coefficients[j] == 0 for j in omitted)
    assert float(engine.measure(state)["Haux"].mid()) == pytest.approx(state.metadata["Haux_eigenvalue"], rel=1e-9)
