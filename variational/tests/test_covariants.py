"""Seeds and the covariant expression language (bfss.covariants)."""

import json
from fractions import Fraction

import pytest

from bfss.algebra import invariants
from bfss.contract import ModularContext, PointEvaluator, primes_1_mod_8
from bfss.covariants import Evaluator, SeedSet
from bfss.fiber import OrbitOperator
from bfss.fock import I, FermionOperator
from bfss.hilbert import Basis, ProjectedGaussian, TrialFunction
from bfss.model import GAMMA, MODEL
from conftest import ROOT
from test_contractions import family_value, flavor_casimir

SEEDS = ("S", "U0", "A", "L2", "O", "H0", "H1", "H2", "H3", "Phi")


@pytest.fixture(scope="module")
def ctx(library):
    return ModularContext(library, primes_1_mod_8(1)[0])


@pytest.fixture(scope="module")
def flavor(ctx, library):
    """The SU(4) Casimir of the flavor index modulo the context's prime."""
    return ctx.image(OrbitOperator(library.space, flavor_casimir()))


@pytest.fixture(scope="module")
def seed_table(engine, seeds):
    chosen = SeedSet([seeds.by_name[n] for n in SEEDS], seeds.fields, seeds.norm)
    functions = [TrialFunction(ProjectedGaussian(2, 1), (0, 0, 0), p, "S") for p in range(3)]
    functions += [TrialFunction(ProjectedGaussian(2, 1), (0, 0, 0), 0, n) for n in SEEDS[1:]]
    basis = Basis(chosen, functions)
    return basis, engine.table(basis)


def test_lagrange_projectors_resolve_the_colour_casimir_on_cubic_sources(ctx, seeds):
    """sum_j Pi_j(C2) = 1 and C2 Pi_j(C2) = j(j+1) Pi_j(C2) on Y^3 S (colour spins j <= 3)."""
    S = seeds.by_name["S"].on_slice(ctx, None)
    ev = lambda src: Evaluator(ctx, {"S": S}.__getitem__)(src)
    zero = lambda v: all(not (a % ctx.p).any() for a in v.terms.values())
    assert zero(ev("Pi_0(C2)*Y(X)^3*S+Pi_1(C2)*Y(X)^3*S+Pi_2(C2)*Y(X)^3*S+Pi_3(C2)*Y(X)^3*S-Y(X)^3*S"))
    for j in range(4):
        assert zero(ev(f"C2*Pi_{j}(C2)*Y(X)^3*S-{j * (j + 1)}*Pi_{j}(C2)*Y(X)^3*S"))
    assert zero(ev("[C2,Y(X)]*S-C2*Y(X)*S"))  # C2 S = 0


def test_expressions_divide_only_by_rational_constants(ctx, seeds):
    """Division by a rational constant is exact; an irrational divisor such as sqrt(2) is rejected rather than
    divided by its rational part (write sqrt(2)*Y(X)^2*S/2 for Y(X)^2*S/sqrt(2))."""
    S = seeds.by_name["S"].on_slice(ctx, None)
    ev = lambda src: Evaluator(ctx, {"S": S}.__getitem__)(src)
    zero = lambda v: all(not (a % ctx.p).any() for a in v.terms.values())
    assert zero(ev("3*Y(X)^2*S/6-Y(X)^2*S/2"))
    for divisor in ("sqrt(2)", "(1+sqrt(2))", "(2*sqrt(2)-sqrt(2))"):
        with pytest.raises(ValueError):
            ev(f"Y(X)^2*S/{divisor}")


def test_slice_formula_of_O_is_the_cubic_three_form():
    """C_Omega(D) = x1 x2 x3 (2N - 24): the slice definition of O is sqrt2 C_Omega S."""
    G123 = GAMMA[0] @ GAMMA[1] @ GAMMA[2]
    C_omega = sum((MODEL.bilinear(a, a, G123) for a in range(3)), FermionOperator()) * I
    assert C_omega == MODEL.number * 2 - 24


@pytest.mark.parametrize("source", ["reference"] + sorted((ROOT / "results").glob("*ansatz.json")),
                         ids=lambda p: getattr(p, "name", p))
def test_seeds_take_values_in_the_fibre(ctx, flavor, seeds, source):
    """Every seed, of the published data and of each committed ansatz, is real and homogeneous, its slice values
    are even under the half-turn about axis 1 (to_sector checks that each partner orbit holds its
    representative's value; it does not check the sign under the half-turn about axis 3), and they are SU(4)
    invariant (the flavor Casimir annihilates them)."""
    if source != "reference":
        seeds = SeedSet.from_json(json.loads(source.read_text())["seeds"])
    for seed, value in zip(seeds.seeds, seeds.evaluate(ctx)):
        assert value.grade == 0 and {sum(m) for m in value.terms} == {seed.degree}, seed.name
        for v in value.terms.values():
            ctx.to_sector(v)
            assert not ctx.reduce(flavor @ v).any(), seed.name


def g(table, basis, a, pa, b, pb, norm):
    """Exact pointwise inner product <Y^pa a, Y^pb b> as a polynomial in s, u, v."""
    j, h = table.index[basis.seeds.index[a], pa], table.index[basis.seeds.index[b], pb]
    ft = table.families["g"]
    N, row = ft.lookup()[(j, h)]
    return {m: c * norm for m, c in zip(ft.basis(N), ft.coefficients(N, row)) if c}


def test_seed_constants_encode_exact_orthogonality(seed_table, seeds):
    """The published constants (3360, 80, 1996/29, 149120/29, ...) make these relations exact."""
    basis, table = seed_table
    pointwise_zero = [("S", 0, "U0", 0), ("S", 0, "A", 0), ("S", 0, "L2", 0), ("S", 1, "H0", 0),
                      ("S", 1, "H3", 0), ("S", 1, "O", 0), ("H0", 0, "H1", 0), ("H1", 0, "H3", 0), ("H2", 0, "H3", 0)]
    for pair in pointwise_zero:
        assert g(table, basis, *pair, seeds.norm) == {}, pair
    # H1 and H2 are orthogonal to Y S under every isotropic Gaussian: E[s^2] : E[u] = 783/4 : 54.
    for h in ("H1", "H2"):
        p = g(table, basis, "S", 1, h, 0, seeds.norm)
        assert set(p) == {(2, 0, 0), (0, 1, 0)} and p[(2, 0, 0)] * Fraction(783, 4) + p[(0, 1, 0)] * 54 == 0


def test_slice_formula_seeds_are_covariant_under_the_weyl_group(seed_table, library):
    """Invariants of slice-defined seeds are symmetric and even in the singular values."""
    basis, table = seed_table
    p = primes_1_mod_8(81)[-1]
    ctx = ModularContext(library, p)
    ev = PointEvaluator(ctx, basis.seeds.evaluate(ctx), table.cores)
    residue = lambda q: q.numerator * pow(q.denominator, -1, p) % p
    for point in [(17, 5, 11), (-5, 11, -17), (11, -17, 5)]:
        values = ev(point)
        for j, h in table.families["g"].lookup():
            assert values["g"][j, h] == residue(family_value(table, "g", j, h, [abs(x) for x in point]))


def test_trial_covariants_are_module_elements_with_tabulated_norms(small_basis, small_table, library):
    """<P, P> of P = s^k u^l v^m Y^p Lambda_sigma on the slice equals (s^k u^l v^m)^2 g from the exact table."""
    p = primes_1_mod_8(1)[0]
    ctx = ModularContext(library, p)
    env = small_basis.seeds.environment(ctx)
    x = (2, 5, 9)
    weight = ctx.sectors[(1, 1)].weight
    residue = lambda q: q.numerator * pow(q.denominator, -1, p) % p
    for f in small_basis.functions[::5]:
        P = f.covariant(small_basis.seeds)
        assert P.degree == f.degree(small_basis.seeds) and {sum(m) for m in P.on_slice(ctx, env).terms} == {P.degree}
        value = sum(pow(x[0], m[0], p) * pow(x[1], m[1], p) * pow(x[2], m[2], p) * ctx.to_sector(v)
                    for m, v in P.on_slice(ctx, env).terms.items()) % p
        j = small_table.index[small_basis.core(f)]
        g = family_value(small_table, "g", j, j, x)
        s, u, v = invariants([c * c for c in x])
        k, l, m = f.invariant
        assert int((weight * value % p) @ value % p) == residue(g * (s**k * u**l * v**m) ** 2)
