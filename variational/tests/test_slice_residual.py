"""The slice residual of analysis/slice_residual.py against the analytic blocks (run with ``pytest --slow``)."""

import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "analysis"))
import radial_tail  # noqa: E402
import slice_residual  # noqa: E402

pytestmark = pytest.mark.slow


def test_profile_and_invariant_jets_are_derivatives():
    """f_{a,b} (also for a < b) and s^k u^l v^m with their g-gradients and g-Hessians, by central differences."""
    g, h = np.array([2.25, 0.64, 0.09]), 1e-5
    jets = [lambda q, a=a, b=b: [j[..., 0] for j in radial_tail.profile(a, b, q[:, None], jet=True)]
            for a, b in [(3.0, 0.5), (2.0, 1.0), (1.0, 2.5)]]
    jets += [lambda q, e=e: slice_residual.invariant_jet(e, q) for e in [(1, 0, 0), (2, 1, 0), (1, 1, 1), (0, 2, 1)]]
    for f in jets:
        value, grad, hess = f(g)
        for a in range(3):
            step = h * np.eye(3)[a]
            plus, minus = f(g + step), f(g - step)
            assert (plus[0] - minus[0]) / (2 * h) == pytest.approx(grad[a], rel=1e-7, abs=1e-12)
            assert (plus[1] - minus[1]) / (2 * h) == pytest.approx(hess[a], rel=1e-6, abs=1e-10)


def test_slice_quadrature_reproduces_norm_energy_and_weighted_residual(two_width_state, tmp_path):
    """norm, E = <H> and <H_aux> of a dilated state with two longitudinal widths, from the Hamiltonian and
    H_aux = (1/16) sum Q R^2 Q applied on the slice (radial Laplacian with its Jacobian, rotation generators off the
    slice, frakB and frakV), against the analytic blocks."""
    from bfss.state import save

    state, exact = two_width_state
    path = tmp_path / "state.json"
    save(state, path)
    points, W, data, K0 = slice_residual.run_points(path, (9.0, 40, 20), ROOT / ".cache")
    total = lambda key: K0 * sum(w * d[key] for w, d in zip(W, data))
    M = total("M")
    assert M == pytest.approx(float(exact["M"].mid()), rel=1e-6)
    assert total("H") / M == pytest.approx(float(exact["E"].mid()), rel=1e-5)
    assert total("Haux") / M == pytest.approx(float(exact["Haux"].mid()), rel=1e-5)

