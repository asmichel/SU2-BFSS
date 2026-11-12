"""The radial density of analysis/radial_tail.py against the analytic blocks (run with ``pytest --slow``)."""

import sys
from fractions import Fraction

import numpy as np
import pytest

from bfss.hilbert import ProjectedGaussian
from conftest import ROOT

sys.path.insert(0, str(ROOT / "analysis"))
import radial_tail  # noqa: E402

pytestmark = pytest.mark.slow


def test_profile_matches_its_series():
    """The Bessel form of f_{a,b} on the slice equals the series of ProjectedGaussian where the series converges."""
    for a, b in [(Fraction(2), Fraction(1)), (Fraction(4), Fraction(1, 8)), (Fraction(3, 2), Fraction(3, 2)),
                 (Fraction(1), Fraction(2)), (Fraction(1, 2), Fraction(3, 2))]:
        g = np.array([[2.25], [0.64], [0.09]])
        assert radial_tail.profile(float(a), float(b), g)[0] == pytest.approx(ProjectedGaussian(a, b)(g[:, 0]), rel=1e-13)


def test_radial_density_integrates_to_the_norm_and_mean_square_radius(two_width_state, library):
    """int p(r) dr = ||psi||^2 and int p(r) (r^2 + <rho^2>_r) dr = <R^2> ||psi||^2 for a dilated state with
    two longitudinal widths: the pointwise assembly, the slice measure and the dilation against the blocks."""
    state, exact = two_width_state
    dens = radial_tail.Density(state, library)
    assert dens.lam == 1.5
    t, w = np.polynomial.legendre.leggauss(80)
    r, w = (t + 1) * 5, w * 5
    p, rho2, _ = radial_tail.shape_integrals(dens, r, 32, radial_tail.jacobian_constant())
    norm = np.sum(w * p)
    assert norm == pytest.approx(float(exact["M"].mid()), rel=1e-8)
    assert np.sum(w * p * (r**2 + rho2)) / norm == pytest.approx(float(exact["R2"].mid()), rel=1e-8)
