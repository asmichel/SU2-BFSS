"""Hilbert-space objects, dilations and state persistence."""

from decimal import Decimal
from fractions import Fraction
from math import isqrt

import numpy as np
import pytest
from flint import fmpq_mat

from bfss.algebra import invariants
from bfss.blocks import BLOCKS, Assembler
from bfss.hilbert import Expansion, ProjectedGaussian
from bfss.scalars import RATIONALS
from bfss.state import load, save
from bfss.variational import KAPPA
from conftest import random_expansion
from quadrature import sphere_octant


def test_profile_series_matches_its_sphere_integral():
    n2, w = sphere_octant(60)
    for (a, b), g in [((3, 1), (0.7, 1.9, 0.2)), ((2, 1), (4.0, 0.1, 2.5)), ((1, 2), (1.0, 3.0, 0.5))]:
        direct = np.sum(w * np.exp(-a * sum(g) / 2 + (a - b) / 2 * (n2.T @ np.array(g))))
        assert ProjectedGaussian(a, b)(g) == pytest.approx(direct, rel=1e-12)


def test_profile_dilation():
    """f_{lambda a, lambda b}(X) = f_{a,b}(sqrt(lambda) X)."""
    f, lam, g = ProjectedGaussian(3, 1), Fraction(5, 2), np.array([0.4, 1.3, 0.9])
    assert f.dilated(lam)(g) == pytest.approx(f(float(lam) * g), rel=1e-13)


def test_dilations_form_a_group(small_basis):
    psi = Expansion(small_basis, [Decimal(j + 1) / 7 for j in range(len(small_basis))])
    a, b = Fraction(9, 4), Fraction(4, 9)
    twice = psi.dilated(a).dilated(b)
    assert twice.basis.functions == psi.basis.functions
    assert all(abs(x - y) < Decimal("1e-40") * abs(y) for x, y in zip(twice.coefficients, psi.coefficients))


def quadratic_form(state, name, table, norm):
    """b^{-1/2} c^T B_reduced c, i.e. <psi, B psi> without the common factor pi^nu (b a rational square)."""
    B = Assembler(state.basis.assembler_trials(table), table, norm).matrix(name)
    x = fmpq_mat([[RATIONALS(c)] for c in state.coefficients])
    total = (x.transpose() * B * x)[0, 0]
    b = state.basis.reference_width()
    root = Fraction(isqrt(b.numerator), isqrt(b.denominator))
    assert root**2 == b
    return Fraction(int(total.numerator), int(total.denominator)) / root


def test_dilated_expansion_is_unitary_and_obeys_dilation_laws(small_basis, small_table, small_seeds):
    """<U psi, B U psi> = mu^{e_B} <psi, B psi>, with U psi assembled directly at the dilated widths."""
    mu = Fraction(9, 4)  # sqrt(mu) = 3/2 keeps every width factor rational
    rng = np.random.default_rng(3)
    psi = Expansion(small_basis, [Decimal(int(v)) / 1000 for v in rng.integers(-1000, 1000, len(small_basis))])
    dilated = psi.dilated(mu)
    for name, e in ((n, b.exponent) for n, b in BLOCKS.items()):
        before = quadratic_form(psi, name, small_table, small_seeds.norm)
        after = quadratic_form(dilated, name, small_table, small_seeds.norm)
        expected = before * Fraction(3, 2) ** int(2 * e)
        assert abs(after - expected) <= Fraction(1, 10**40) * (abs(expected) + abs(before)), name


def test_state_file_round_trip(tmp_path, small_basis):
    psi = Expansion(small_basis.dilated(Fraction(3, 2)), [Decimal(j + 1) / 7 for j in range(len(small_basis))],
                    {"note": "round trip"})
    save(psi, tmp_path / "state.json")
    back = load(tmp_path / "state.json")
    assert back.basis.functions == psi.basis.functions
    assert back.basis.seeds.to_json() == psi.basis.seeds.to_json()
    assert back.coefficients == psi.coefficients
    assert back.metadata == psi.metadata


def test_ansatz_reproduces_the_published_ordering(reference_source):
    import json

    from bfss.ansatz import Ansatz
    from bfss.reference import reference_state

    basis = reference_state(reference_source).basis
    canonical = basis.dilated(1 / basis.reference_width())
    ansatz = Ansatz.of(canonical)
    assert ansatz.basis().functions == canonical.functions
    again = Ansatz.from_json(json.loads(json.dumps(ansatz.to_json())))
    assert again.basis().functions == canonical.functions


def test_command_line_measure_and_optimize(tmp_path, small_basis, small_table, capsys):
    from bfss.cli import main
    from conftest import CACHE

    psi = Expansion(small_basis, [Decimal(1)] + [Decimal(0)] * (len(small_basis) - 1))
    save(psi, tmp_path / "psi.json")
    main(["--cache", str(CACHE), "measure", str(tmp_path / "psi.json")])
    assert "E =" in capsys.readouterr().out
    main(["--cache", str(CACHE), "--prec", "200", "optimize", str(tmp_path / "psi.json"), "-o", str(tmp_path / "opt.json")])
    out = capsys.readouterr().out
    best = load(tmp_path / "opt.json")
    measured = {line.split("=")[0].strip(): line.split("=")[1].split("+/-")[0] for line in out.splitlines() if "=" in line}
    # The measured <H_aux> of the saved state is the optimizer's smallest eigenvalue.
    assert float(measured["Haux"]) == pytest.approx(best.metadata["Haux_eigenvalue"], rel=1e-9)
    assert best.metadata["fidelity_lower_bound"] == max(0.0, 1 - KAPPA * best.metadata["Haux_eigenvalue"])
    # At a given dilation (a fraction) the state is placed exactly there, and <H_aux> is larger than at the optimum.
    lam = Fraction(best.metadata["dilation"]) * Fraction(5, 4)
    main(["--cache", str(CACHE), "--prec", "200", "optimize", str(tmp_path / "psi.json"), "-o", str(tmp_path / "at.json"),
          "--dilation", str(lam)])
    fixed = load(tmp_path / "at.json")
    assert Fraction(fixed.metadata["dilation"]) == lam
    assert fixed.metadata["Haux_eigenvalue"] > best.metadata["Haux_eigenvalue"]
    # A measurement can store the observables, as the optimizer does after its optimum.
    capsys.readouterr()
    main(["--cache", str(CACHE), "--prec", "200", "measure", "--store", str(tmp_path / "at.json")])
    printed = {line.split("=")[0].strip(): line.split("=")[1].split("+/-")[0].strip()
               for line in capsys.readouterr().out.splitlines() if "=" in line}
    stored = load(tmp_path / "at.json").metadata["observables"]
    assert stored["Haux"] == printed["Haux"] and stored["E"] == printed["E"] and stored == fixed.metadata["observables"]


def test_state_is_assembled_pointwise_from_its_building_blocks(small_basis, small_table, library):
    """|psi(D(x))|^2 from seeds, profiles and coefficients equals sum c_j c_h f_j f_h w_j w_h g_jh, with w_j the
    invariant prefactor s^k u^l v^m of function j."""
    from bfss.fiber import ComplexContext
    from test_contractions import family_value

    psi = random_expansion(small_basis, 5)
    ctx = ComplexContext(library.ops, library.space)
    x = (Fraction(7, 10), Fraction(13, 10), Fraction(1, 2))
    vec = psi.on_slice(x, ctx)
    density = ctx.inner(vec, vec).real * float(small_basis.seeds.norm)
    g = [float(c) ** 2 for c in x]
    s, u, v = invariants(g)
    F = [float(c) * f.profile(g) * s ** f.invariant[0] * u ** f.invariant[1] * v ** f.invariant[2]
         for c, f in zip(psi.coefficients, small_basis.functions)]
    idx = [small_table.index[small_basis.core(f)] for f in small_basis.functions]
    expected = sum(F[j] * F[h] * float(family_value(small_table, "g", idx[j], idx[h], x) * small_basis.seeds.norm)
                   for j in range(len(F)) for h in range(len(F)) if (idx[j], idx[h]) in small_table.families["g"].lookup())
    assert density == pytest.approx(expected, rel=1e-10)


@pytest.mark.slow
def test_optimum_over_unequal_longitudinal_widths_is_normalized(engine, small_seeds, small_table):
    """With profiles of two longitudinal widths the optimum is a unit vector by an independent quadrature of
    its physical coefficients, and measuring it reproduces the optimizer's eigenvalue."""
    from math import pi, sqrt

    from bfss.hilbert import Basis, TrialFunction
    from conftest import SMALL_CORES
    from test_blocks import SliceIntegrands

    widths = ((Fraction(2), Fraction(1)), (Fraction(3, 2), Fraction(1, 2)))
    basis = Basis(small_seeds, [TrialFunction(ProjectedGaussian(a, b), (0, 0, 0), c.power, small_seeds.seeds[c.seed].name)
                                for a, b in widths for c in SMALL_CORES])
    state = engine.optimize(basis, nb=8)
    assert float(engine.measure(state)["Haux"].mid()) == pytest.approx(state.metadata["Haux_eigenvalue"], rel=1e-9)
    canonical = state.dilated(1 / Fraction(state.metadata["dilation"]))
    T, c = canonical.basis.assembler_trials(small_table), [float(x) for x in canonical.coefficients]
    oracle = SliceIntegrands(small_table, small_seeds.norm, widths)
    norm = sum(c[j] * c[h] * oracle.element("M", T[j], T[h]) / sqrt(float(T[j][1] + T[h][1]) / 2)
               for j in range(len(T)) for h in range(len(T)))
    assert norm * pi ** 13.5 == pytest.approx(1, rel=1e-5)


def test_cached_tables_serve_any_subset_of_their_cores(engine, small_basis, small_table):
    """Exact contraction coefficients do not depend on the other cores of a table, so a cached table
    serves every basis whose cores it contains (here instead of minutes of new contractions)."""
    from bfss.hilbert import Basis

    subset = Basis(small_basis.seeds, [f for f in small_basis.functions if f.power == 0])
    assert len(subset.cores()) < len(small_basis.cores())
    assert engine.table(subset).cores == small_table.cores


@pytest.mark.parametrize("ansatz, state", [("ansatz.json", "optimized_state.json"),
                                           ("extended_ansatz.json", "extended_state.json"),
                                           ("seeded_ansatz.json", "seeded_state.json")])
def test_committed_states_expand_over_their_ansatzes(ansatz, state):
    """Each committed state expands over exactly the basis of its committed ansatz, dilated by its λ*."""
    import json

    from bfss.ansatz import Ansatz
    from conftest import ROOT

    basis = Ansatz.from_json(json.loads((ROOT / "results" / ansatz).read_text())).basis()
    psi = load(ROOT / "results" / state)
    canonical = basis.dilated(1 / basis.reference_width())
    assert psi.basis.functions == canonical.dilated(Fraction(psi.metadata["dilation"])).functions
    assert psi.metadata["fidelity_lower_bound"] == max(0.0, 1 - KAPPA * psi.metadata["Haux_eigenvalue"])
