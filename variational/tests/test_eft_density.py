"""Independent quadrature and dilation checks without constructing the fermion library."""

import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "analysis"))
import eft_density  # noqa: E402


class GaussianDensity:
    """Canonical |psi|²=exp(-|X|²), with the same dilation contract as Density."""

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, x):
        return np.exp(-np.sum(x**2, axis=0))


K0 = 1024 * np.pi**13 / 4725


@pytest.mark.parametrize("lam", [0.7, 2.5])
def test_joint_integrates_to_independent_triangular_moments(lam):
    """Polar/chamber bounds and two density delta Jacobians match a Cartesian rule."""
    dens = GaussianDensity(lam)
    t, w = np.polynomial.legendre.leggauss(64)
    for r in (0.3, 1.0, 2.0, 5.6):
        # Split at r, where the angular lower bound switches branches.
        rho = np.concatenate([(t + 1) * r / 2,
                              r + (t + 1) * (np.sqrt(2) - 1) * r / 2])
        weights = np.concatenate([w * r / 2, w * (np.sqrt(2) - 1) * r / 2])
        p = eft_density.joint_density(dens, [r], rho, 48, K0)[0]
        moments = eft_density.shape_moments(dens, [r], 48, K0)
        assert np.sum(weights * p) == pytest.approx(moments["p"][0], rel=2e-11)
        for power in (2, 4):
            measured = np.sum(weights * p * rho**power) / np.sum(weights * p)
            assert measured == pytest.approx(moments[f"rho{power}"][0], rel=2e-11)


def test_gaussian_radial_normalization_and_mean_square_radius():
    """Absolute angular normalization and physical dilation against 27 Gaussian coordinates."""
    lam = 1.7
    dens = GaussianDensity(lam)
    t, w = np.polynomial.legendre.leggauss(80)
    r, weights = (t + 1) * 5, w * 5
    moments = eft_density.shape_moments(dens, r, 32, K0)
    norm = np.sum(weights * moments["p"])
    assert norm == pytest.approx(np.pi**13.5, rel=1e-11)
    mean_square = np.sum(weights * moments["p"] * (r**2 + moments["rho2"])) / norm
    assert mean_square == pytest.approx(27 / (2 * lam), rel=1e-11)


def test_joint_support_and_conditional_export():
    dens = GaussianDensity(2.5)
    r = 1.2
    rho = np.array([0, 0.5, r, np.sqrt(2) * r, 2 * r])
    section = eft_density.evaluate(dens, K0, "conditional", [r], rho, orders=[24, 32])
    p = np.asarray(section["p"])
    assert np.array_equal(p[[0, 3, 4]], np.zeros(3))
    assert np.all(p[[1, 2]] > 0)
    assert section["p_cond"] == pytest.approx(p / section["p_radial"])
    assert section["rho2_over_eft"] == pytest.approx(r * section["rho2"] / (4 * np.sqrt(2)))
    assert section["rho4_over_rho2_sq"] == pytest.approx(section["rho4"] / section["rho2"]**2)
    assert section["order_agreement"]["24/32"]["p"]["max_rel"] < 1e-11
