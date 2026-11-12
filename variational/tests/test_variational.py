"""Orthonormal coordinates and the variational problem (the paper's Eq. (22))."""

from dataclasses import replace
from fractions import Fraction

import flint
import numpy as np
import pytest
from flint import arb

from bfss.blocks import OBSERVABLES
from bfss.hilbert import Basis
from bfss.scalars import Reals
from bfss.variational import (KAPPA, ReducedBlocks, gram_schmidt, lowest_eigenpair, optimize, orthonormal_frame,
                              to_float)


@pytest.fixture(autouse=True)
def precision():
    with flint.ctx.workprec(256):
        yield


@pytest.fixture(scope="module")
def blocks(small_basis, small_table):
    return ReducedBlocks(small_basis, small_table)


@pytest.mark.parametrize("block", [2048, 3])
def test_gram_schmidt_frame_is_orthonormal(blocks, block):
    """L^{-1} M L^{-T} = 1, also with the products by inverse factors in fixed point (block 3: every product of
    at least 9 rows)."""
    subset = blocks.independent_subset()
    M = blocks.arb("M", subset)
    Linv, kept = gram_schmidt(M, nb=8, block=block)
    assert kept == list(range(len(subset)))
    residual = to_float(Linv * M * Linv.transpose()) - np.eye(len(subset))
    assert np.abs(residual).max() < 1e-60
    assert to_float(Linv)[np.triu_indices(len(subset), 1)].max() == 0  # ordered Gram-Schmidt


def test_dependent_functions_are_omitted(small_basis, small_table):
    """A zero Gram-Schmidt residual is detected at its position in the order: exactly by the rank
    profile, and with ball arithmetic by the relative tolerance of the orthonormalization."""
    f = small_basis.functions
    doubled = Basis(small_basis.seeds, f[:5] + [f[2]] + f[5:])
    expected = [j for j in range(len(f) + 1) if j != 5]
    assert ReducedBlocks(doubled, small_table).independent_subset() == expected
    assert orthonormal_frame(ReducedBlocks(doubled, small_table, Reals(256)), nb=8).subset == expected


def rayleigh(blocks, x, lam):
    haux = OBSERVABLES["Haux"]
    return haux.dilated({k: blocks.quadratic_form(k, x) for k in haux.parts}, arb(lam)) / blocks.quadratic_form("M", x)


def test_optimum_is_the_minimal_rayleigh_quotient(blocks, small_basis, small_table):
    result = optimize(blocks, nb=8)
    assert float(rayleigh(blocks, result.x, result.dilation).mid()) == pytest.approx(result.eigenvalue, rel=1e-12)
    assert result.fidelity_bound == max(0.0, 1 - KAPPA * result.eigenvalue)
    # Variational principle: perturbing the optimum or shrinking the space cannot lower <H_aux>.
    x = list(result.x)
    x[3] = x[3] + arb("1e-3") * abs(x[3]) + arb("1e-9")
    assert float(rayleigh(blocks, x, result.dilation).mid()) > result.eigenvalue
    smaller = ReducedBlocks(Basis(small_basis.seeds, small_basis.functions[:-6]), small_table)
    assert optimize(smaller, nb=8).eigenvalue > result.eigenvalue
    # Ball matrix elements give the same optimum as exact ones.
    balls = optimize(ReducedBlocks(small_basis, small_table, Reals(256)), nb=8)
    assert balls.eigenvalue == pytest.approx(result.eigenvalue, rel=1e-12)
    assert balls.dilation == pytest.approx(result.dilation, rel=1e-9)



def test_fixed_dilation_transforms_one_block(blocks):
    """U_lambda^-1 H_aux U_lambda transformed as one block has the smallest eigenvalue of the dilated transformed
    parts, also with functions left out of the frame: at the searched optimal dilation the same optimum, at another
    dilation a larger eigenvalue, the Rayleigh quotient of its coefficients there. A dilation is positive."""
    searched = optimize(blocks, nb=8)
    lam = Fraction(f"{searched.dilation:.16g}")
    assert optimize(blocks, nb=8, dilation=lam).eigenvalue == pytest.approx(searched.eigenvalue, rel=1e-10)
    other = optimize(blocks, nb=8, omit=(3, 10), dilation=lam * Fraction(5, 4))
    frame, haux = orthonormal_frame(blocks, nb=8, omit=(3, 10)), OBSERVABLES["Haux"]
    parts = {k: frame.transform(blocks.arb(k, frame.subset)) for k in haux.parts}
    assert other.eigenvalue == pytest.approx(lowest_eigenpair(haux.dilated(parts, other.dilation))[0], rel=1e-12)
    assert other.dilation == float(lam * Fraction(5, 4)) and other.eigenvalue > searched.eigenvalue
    assert float(rayleigh(blocks, other.x, other.dilation).mid()) == pytest.approx(other.eigenvalue, rel=1e-12)
    with pytest.raises(ValueError):
        optimize(blocks, nb=8, dilation=Fraction(0))


@pytest.mark.parametrize("block", [2048, 5])
def test_fixed_point_transform_matches_ball_products(blocks, block):
    """Blocks in the orthonormal frame, from equilibrated fixed-point integer products, agree with the
    product of ball matrices to float64 accuracy, in one product and block by block (the last block ragged)."""
    frame = replace(orthonormal_frame(blocks, nb=8), block=block)
    assert block > len(frame.subset) or len(frame.subset) % block
    for name in ("Haux_T", "Haux_u", "Haux_Y", "T"):
        B = blocks.arb(name, frame.subset)
        balls = to_float(frame.Linv * B * frame.Linv.transpose())
        assert np.abs(frame.transform(B) - balls).max() <= 1e-14 * np.abs(balls).max(), name
