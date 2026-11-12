"""Checks of the weight-coordinate machinery against the Cartesian Weyl-Clifford algebra (kinematics.py).

  python tests/check_pairing.py
"""
import random
import sys
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bfss import basis as B, pairing as Pa
from bfss.kinematics import star as cstar
from check_problem import explicit, project

rng = random.Random(3)


def rand_poly(nx, np_, nf, terms=3):
    out = {}
    for _ in range(terms):
        xs = tuple(sorted(rng.randrange(B.NX) for _ in range(nx)))
        ps = tuple(sorted(rng.randrange(B.NX) for _ in range(np_)))
        fs = tuple(sorted(rng.sample(range(B.NF), nf)))
        out[(xs, ps, fs)] = (Fr(rng.randint(-3, 3)), Fr(rng.randint(-3, 3)))
    return {k: v for k, v in out.items() if v != B.Z0}


# 1. conversions are inverse, star and dagger agree with the Cartesian ones
for shape in [(1, 1, 0), (0, 0, 2), (2, 1, 1), (0, 1, 3), (1, 2, 2)]:
    f, g = rand_poly(*shape), rand_poly(*reversed(shape))
    F, G_ = B.to_cartesian(f), B.to_cartesian(g)
    assert B.from_cartesian(F)[0] == f, "from_cartesian(to_cartesian) != id"
    assert B.to_cartesian(Pa.star(f, g)) == cstar(F, G_), f"star mismatch {shape}"
    assert B.to_cartesian(Pa.dagger(f)) == F.dagger(), f"dagger mismatch {shape}"
print("conversions, star, dagger: ok")

# 2. fast pairings (Pairing.evaluate) against explicit products
lam = (2, 2, 0, 0, 0)
mult = B.multiplicities(6)
hws = []
for (g, l), m in mult.items():
    if l == lam:
        s = B.sector(g, l, m)
        hws += [s.poly(i) for i in range(len(s.rows))]
irr = Pa.Irrep(lam, min(hws, key=len))
comps = [irr.components(h) for h in hws]
fun = Pa.Functionals(Pa.primes(1)[0], 16, 5)
G = Pa.Pairing([[Pa.dagger(o) for o in c] for c in comps], comps, irr).evaluate(fun)
for a in range(len(hws)):
    for b in range(len(hws)):
        assert (project(explicit(irr, comps[a], comps[b]), fun) == G[a, b]).all(), f"pairing mismatch {a} {b}"
print(f"pairings of {len(hws)} copies of {lam}: ok")

# 3. the pairing is invariant: annihilated by every simple raising and lowering derivation
act = B.action()
for a, b in [(0, 1), (2, 2), (1, 3)]:
    P = explicit(irr, comps[a], comps[b])
    assert P and all(not B.derive(P, E) for E in act["raise"] + act["lower"]), "pairing not invariant"
print("pairings are invariant: ok")

# 4. slicing the functionals changes nothing
fun2 = Pa.Functionals(Pa.primes(1)[0], 40, 5)
P = Pa.Pairing([[Pa.dagger(o) for o in c] for c in comps], comps, irr)
full = P.evaluate(fun2)
Pa.SLICE = 7 * max(len(P.L.slots), len(P.R.slots)) * len(comps)
assert (P.evaluate(fun2) == full).all(), "sliced evaluation differs"
print("sliced functionals: ok")

# 5. null multiplets X * N are highest-weight vectors (Clebsch-Gordan), at level 8
from bfss import moments as Mo
h8 = Mo.canonical(8, 4, log=lambda s: None)
bad = sum(1 for lam, parts in h8.nulls for part in parts for E in act["raise"] if part and B.derive(part, E))
print(f"{len(h8.nulls)} null multiplets at level 8, {len(h8.ward)} Ward forms; not highest weight: {bad}")
assert not bad
