"""Declarative variational ansatz: ordered shells of trial wavefunctions.

A shell lists profile widths, invariant prefactors s^k u^l v^m and, for each seed, a
range of Yukawa powers; it expands (widths outermost, then prefactors, then seeds in
the listed order, then powers) into trial wavefunctions f_{a,b} s^k u^l v^m Y^p Lambda_sigma (the
paper's Eq. (19)).  The order of the expansion is the Gram-Schmidt order.  Seeds come from a
``SeedSet``, so both the polynomial factors and the basis are data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .covariants import SeedSet
from .hilbert import Basis, ProjectedGaussian, TrialFunction


@dataclass
class Shell:
    widths: list[tuple[Fraction, Fraction]]
    invariants: list[tuple[int, int, int]]
    powers: dict[str, tuple[int, int]]  # seed -> inclusive power range, in expansion order

    def functions(self):
        return [TrialFunction(ProjectedGaussian(a, b), tuple(inv), p, seed)
                for a, b in self.widths for inv in self.invariants
                for seed, (lo, hi) in self.powers.items() for p in range(lo, hi + 1)]

    def to_json(self):
        return {"widths": [[str(a), str(b)] for a, b in self.widths], "invariants": [list(i) for i in self.invariants],
                "powers": {k: list(v) for k, v in self.powers.items()}}

    @classmethod
    def from_json(cls, d):
        return cls([(Fraction(a), Fraction(b)) for a, b in d["widths"]], [tuple(i) for i in d["invariants"]],
                   {k: tuple(v) for k, v in d["powers"].items()})


@dataclass
class Ansatz:
    seeds: SeedSet
    shells: list[Shell] = field(default_factory=list)

    def basis(self) -> Basis:
        return Basis(self.seeds, [f for shell in self.shells for f in shell.functions()])

    def to_json(self):
        return {"schema": "bfss.ansatz.v1", "seeds": self.seeds.to_json(), "shells": [s.to_json() for s in self.shells]}

    @classmethod
    def from_json(cls, d):
        return cls(SeedSet.from_json(d["seeds"]), [Shell.from_json(s) for s in d["shells"]])

    @classmethod
    def of(cls, basis: Basis) -> Ansatz:
        """Shell description of an ordered basis: each run of one width and one prefactor is a shell (split where a
        seed recurs out of order), consecutive powers of a seed a range; not the shortest description."""
        shells = []
        for f in basis.functions:
            key = ((f.profile.a, f.profile.b), f.invariant)
            last = shells[-1] if shells else None
            if last and [last.widths[-1], last.invariants[-1]] == list(key) and len(last.widths) == len(last.invariants) == 1:
                lo, hi = last.powers.get(f.seed, (f.power, f.power - 1))
                if f.seed in last.powers and f.power == hi + 1 and list(last.powers)[-1] == f.seed:
                    last.powers[f.seed] = (lo, f.power)
                    continue
                if f.seed not in last.powers:
                    last.powers[f.seed] = (f.power, f.power)
                    continue
            shells.append(Shell([key[0]], [key[1]], {f.seed: (f.power, f.power)}))
        return cls(basis.seeds, shells)
