"""Elements of the singlet Hilbert space and the trial ansatz.

The physical space is the gauge- and Spin(9)-invariant subspace of
L^2(R^{3x9}, d^27X; F).  ``HilbertVector`` is the abstract element; the ansatz
supplies the concrete ``TrialFunction`` f_{a,b}(X) s^k u^l v^m Y(X)^p Lambda_sigma(X), and an
``Expansion`` is a generic finite linear combination over an ordered ``Basis``.
``Expansion.dilated`` applies the unitary dilation (U_lambda psi)(X) = lambda^{nu/2} psi(sqrt(lambda) X);
``TrialFunction.dilated`` rescales the profile only, without the factor lambda^{(nu+d)/2}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from .covariants import InvariantMultiple, SeedSet, YukawaPower
from .model import NU


class HilbertVector(ABC):
    """Element of the singlet subspace of L^2(R^27; F)."""

    @abstractmethod
    def dilated(self, lam: Fraction) -> HilbertVector:
        """The vector at widths scaled by lambda: U_lambda applied with its factor (Expansion) or without it
        (TrialFunction)."""


class Profile(ABC):
    """Rotation-invariant positive profile f(G) multiplying a polynomial factor."""

    @abstractmethod
    def dilated(self, lam: Fraction) -> Profile:
        """Profile of f(sqrt(lambda) X)."""

    @abstractmethod
    def __call__(self, g) -> float:
        """Value at a configuration with eigenvalues g of G = X X^T."""

    @abstractmethod
    def to_json(self) -> dict:
        """Definition as data, with a "kind" key naming the profile family."""

    @staticmethod
    def from_json(data) -> Profile:
        return PROFILES[data["kind"]].from_json(data)


@dataclass(frozen=True)
class ProjectedGaussian(Profile):
    """f_{a,b}(X) = (1/4 pi) int_{S^2} exp(-a s/2 + (a-b)/2 n^T G n) dOmega(n)  (the paper's Eq. (18))."""

    a: Fraction
    b: Fraction

    def __post_init__(self):
        object.__setattr__(self, "a", Fraction(self.a))
        object.__setattr__(self, "b", Fraction(self.b))
        assert self.a > 0 and self.b > 0

    def dilated(self, lam):
        return ProjectedGaussian(self.a * lam, self.b * lam)

    def to_json(self):
        return {"kind": "projected_gaussian", "a": str(self.a), "b": str(self.b)}

    @classmethod
    def from_json(cls, data):
        return cls(Fraction(data["a"]), Fraction(data["b"]))

    def __call__(self, g, terms=80):
        """Absolutely convergent series in the eigenvalues of G."""
        eta = float(self.a - self.b) / 2
        m = np.arange(terms)
        half = np.concatenate([[1.0], np.cumprod(0.5 + m[:-1])])  # (1/2)_m
        fact = np.concatenate([[1.0], np.cumprod(m[1:].astype(float))])  # m!
        series = [half * (eta * x) ** m / fact for x in g]
        total = np.convolve(np.convolve(series[0], series[1]), series[2])[:terms]
        three_halves = np.concatenate([[1.0], np.cumprod(1.5 + m[:-1])])  # (3/2)_M
        return float(np.exp(-float(self.a) * sum(g) / 2) * (total / three_halves).sum())


PROFILES = {"projected_gaussian": ProjectedGaussian}


@dataclass(frozen=True)
class TrialFunction(HilbertVector):
    """f_{a,b}(X) s^k u^l v^m Y(X)^p Lambda_sigma(X): a trial wavefunction psi_j (the paper's Eq. (19))."""

    profile: Profile
    invariant: tuple[int, int, int]
    power: int
    seed: str

    def covariant(self, seeds: SeedSet) -> InvariantMultiple:
        """The polynomial factor s^k u^l v^m Y^p Lambda_sigma (the paper's Eq. (17)) in the covariant module."""
        return InvariantMultiple(self.invariant, YukawaPower(seeds.by_name[self.seed], self.power))

    def degree(self, seeds: SeedSet) -> int:
        return self.covariant(seeds).degree

    def dilated(self, lam):
        """Dilated profile, same polynomial, no factor: U_lambda psi_j = lambda^{(nu+d)/2} psi_j(dilated)."""
        return replace(self, profile=self.profile.dilated(lam))

    def to_json(self):
        return {"seed": self.seed, "power": self.power, "invariant": list(self.invariant),
                "profile": self.profile.to_json()}

    @classmethod
    def from_json(cls, data):
        return cls(Profile.from_json(data["profile"]), tuple(data["invariant"]), data["power"], data["seed"])


@dataclass
class Basis:
    """Ordered trial wavefunctions over a seed set."""

    seeds: SeedSet
    functions: list[TrialFunction]

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, i):
        return self.functions[i]

    def degrees(self):
        return [f.degree(self.seeds) for f in self.functions]

    def dilated(self, lam):
        return Basis(self.seeds, [f.dilated(lam) for f in self.functions])

    def core(self, f: TrialFunction):
        """(seed index, power) of the core Y^p Lambda_sigma of a trial wavefunction."""
        return self.seeds.index[f.seed], f.power

    def cores(self):
        """Distinct cores, in a canonical order."""
        return sorted({self.core(f) for f in self.functions})

    def assembler_trials(self, table):
        """(a, b, invariant exponents, position of the core in a contraction table) for the block assembler."""
        return [(f.profile.a, f.profile.b, f.invariant, table.index[self.core(f)]) for f in self.functions]

    def reference_width(self):
        """Largest longitudinal width: the canonical basis, whose reduced blocks are assembled, has it at 1."""
        return max(f.profile.b for f in self.functions)


@dataclass
class Expansion(HilbertVector):
    """Finite linear combination sum_j c_j psi_j with high-precision real coefficients."""

    basis: Basis
    coefficients: list[Decimal]
    metadata: dict = field(default_factory=dict)

    def dilated(self, lam, digits=60):
        """U_lambda psi_j = lambda^{(nu + d_j)/2} psi_j(lambda widths), with nu = 27/2 (to ``digits`` digits)."""
        lam = Fraction(lam)
        with localcontext() as ctx:
            ctx.prec = digits
            quarter = (Decimal(lam.numerator) / Decimal(lam.denominator)).sqrt().sqrt()
            scaled = [c * quarter ** int(2 * NU + 2 * d) if c else c
                      for c, d in zip(self.coefficients, self.basis.degrees())]
        return Expansion(self.basis.dilated(lam), scaled, dict(self.metadata))

    def on_slice(self, x, ctx):
        """psi(D(x)) in orbit coordinates: sum_j c_j f_j s^k u^l v^m Y(x)^p Lambda_sigma(x), assembled from the seeds.

        ``ctx`` is a ``fiber.SliceArithmetic`` (e.g. ``ComplexContext``); <psi, psi>(x) is
        ``ctx.inner(v, v)`` times the seed normalization.
        """
        env, g = self.basis.seeds.environment(ctx), [float(c) ** 2 for c in x]
        total = 0
        for c, f in zip(self.coefficients, self.basis.functions):
            P = f.covariant(self.basis.seeds).on_slice(ctx, env)
            value = sum(float(x[0]) ** m[0] * float(x[1]) ** m[1] * float(x[2]) ** m[2] * v for m, v in P.terms.items())
            total = total + float(c) * f.profile(g) * value
        return total
