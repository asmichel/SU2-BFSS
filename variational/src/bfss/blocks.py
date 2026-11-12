"""Observables as quadratic forms between trial wavefunctions, assembled in a number system.

For trial wavefunctions psi = f_{a,b} w core (w = s^k u^l v^m; core = Y^p Lambda_sigma)
the pointwise integrands of the matrix elements are rewritten through integration
by parts into three kinds of pieces, each built from the contraction families of
``contract``:

  Insertion(q, family):  w_j w_h q(s, u, v) <core_j, core_h>_family
  Gradient(q):           (q/2) sum_ai <d_ai psi_j, d_ai psi_h>
  OneSidedB(c):          c <psi_j, frakB psi_h>,  frakB = sum X_ai B_{ai,bk} d_bk (Hermitian)

so that M, u, R2 to R8 (s to s^4), Haux_u, Y are insertions, T = Gradient(1),
Haux_T = Gradient(s) - (1/8) frakB and Haux_Y = s Y - (1/8) i frakV (frakV = sum_alpha c_alpha V_alpha, an
insertion of the family v).  Each block scales homogeneously under the dilation U_lambda, and an observable is
a sum of blocks: U_lambda^-1 H U_lambda = lambda T + lambda^-2 u + lambda^-1/2 Y and
U_lambda^-1 H_aux U_lambda = Haux_T + lambda^-3 Haux_u + lambda^-3/2 Haux_Y; the three parts of H_aux scale
like T, u and Y times s.  With oriented profiles the derivative of a profile is -(A_j(n) X) f_{j;n},
A_j = a_j I - 2 eta_j n n^T, which produces the orientation atoms of ``gaussian.ATOMS``.  A term (family,
transposed, component, atom, q) contributes sum_mu phi_mu sum_rho q_rho master(mu + rho, atom), evaluated
exactly or in ball arithmetic of a working precision (``scalars``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np

from .algebra import E1 as S
from .algebra import E2 as U
from .algebra import GENERATOR as G
from .algebra import MatrixPolynomial, Polynomial
from .contract import FAMILY_SHAPE
from .gaussian import OrientationIntegrals
from .scalars import RATIONALS, Scalars

ONE = Polynomial.const(1)
ADJ_G = G * G - S * G + U  # adjugate of G by Cayley-Hamilton (s, u, v are the invariants of G)


class TrialData:
    """Scalar data of a trial wavefunction in the integrand: w = s^k u^l v^m, its gradient matrix, widths."""

    def __init__(self, a, b, exponents):
        self.a, self.eta = Fraction(a), (Fraction(a) - Fraction(b)) / 2
        self.w = Polynomial.monomial(exponents)
        ws, wu, wv = (self.w.derivative(i) for i in range(3))
        # L = 2 [w_s I + w_u (s I - G) + w_v adj G] - a w
        self.L = 2 * (ws + wu * (S - G) + wv * ADJ_G) - self.a * self.w


def _contract_atoms(F: MatrixPolynomial, side, coef):
    """x^T F x for x = n or m as {atom: polynomial}."""
    a, b, c = F
    return {"1": coef * a, side + "1": coef * b, side + "2": coef * c}


def _trace_terms(L: MatrixPolynomial, coef):
    """tr(L F) for F = sum_b phi_b G^b: structure polynomial for each component b."""
    return {b: coef * (L * G**b).trace() for b in range(3)}


class Piece(ABC):
    """Part of the pointwise integrand of a quadratic form between psi_j and psi_h."""

    @abstractmethod
    def terms(self, tj: TrialData, th: TrialData) -> list:
        """(family, transposed, component, atom, structure polynomial) contributions."""


@dataclass
class Insertion(Piece):
    """Multiplication by an invariant q(s, u, v), paired through a contraction family (g, y or v)."""

    family: str
    weight: Polynomial

    def terms(self, tj, th):
        return [(self.family, False, 0, "1", tj.w * th.w * self.weight)]


@dataclass
class Gradient(Piece):
    """(q/2) sum_ai <d_ai psi_j, d_ai psi_h> with d_ai (f w core) = f [(Z X)_ai core + w d_ai core]."""

    weight: Polynomial

    def terms(self, tj, th):
        half, w = self.weight / 2, tj.w * th.w
        # tr(Z_j G Z_h) g with Z = L + 2 eta w x x^T, x = n (trial j) or m (trial h)
        LjG, GLh = tj.L * G, G * th.L
        terms = [("g", False, 0, "1", half * (LjG * th.L).trace())]
        terms += [("g", False, 0, atom, q) for atom, q in _contract_atoms(LjG, "m", 2 * th.eta * half * th.w).items()]
        terms += [("g", False, 0, atom, q) for atom, q in _contract_atoms(GLh, "n", 2 * tj.eta * half * tj.w).items()]
        terms.append(("g", False, 0, "z1", 4 * tj.eta * th.eta * half * w))
        # w_h tr(Z_j N_jh) + w_j tr(Z_h N_hj), N_jh the lift of nu_jh
        for transposed, own, other, side in ((False, tj, th, "n"), (True, th, tj, "m")):
            terms += [("nu", transposed, b, "1", q) for b, q in _trace_terms(own.L, half * other.w).items()]
            terms += [("nu", transposed, b, atom, 2 * own.eta * half * w)
                      for b, atom in enumerate(("1", side + "1", side + "2"))]
        return terms + [("tau", False, 0, "1", half * w)]


@dataclass
class OneSidedB(Piece):
    """c <psi_j, frakB psi_h> = c w_j [tr(Z_h T_jh) + w_h beta_jh], T_jh the lift of t_jh."""

    coefficient: Fraction

    def terms(self, tj, th):
        c, w = self.coefficient, tj.w * th.w
        terms = [("t", False, b, "1", q) for b, q in _trace_terms(th.L, c * tj.w).items()]
        terms += [("t", False, b, atom, 2 * c * th.eta * w) for b, atom in enumerate(("1", "m1", "m2"))]
        return terms + [("beta", False, 0, "1", c * w)]


class QuadraticForm(ABC):
    """Hermitian form on the singlet space with U_lambda^-1 Q U_lambda = sum_B lambda^{e_B} B."""

    name: str

    @property
    @abstractmethod
    def parts(self) -> dict:
        """Homogeneous blocks and their dilation exponents e_B."""

    def dilated(self, blocks: dict, lam):
        """U_lambda^-1 Q U_lambda from the values (numbers or matrices) of its blocks."""
        return sum(lam**e * blocks[b] for b, e in self.parts.items())


@dataclass
class Block(QuadraticForm):
    """Dilation-homogeneous component of an observable, defined by its integrand pieces."""

    name: str
    exponent: Fraction
    pieces: tuple

    @property
    def parts(self):
        return {self.name: self.exponent}

    def terms(self, tj, th):
        return [t for p in self.pieces for t in p.terms(tj, th) if t[4]]


@dataclass
class Observable(QuadraticForm):
    name: str
    blocks: tuple

    @property
    def parts(self):
        return {b.name: b.exponent for b in self.blocks}


BLOCKS = {b.name: b for b in [
    Block("M", Fraction(0), (Insertion("g", ONE),)),
    Block("T", Fraction(1), (Gradient(ONE),)),
    Block("u", Fraction(-2), (Insertion("g", U),)),
    Block("Y", Fraction(-1, 2), (Insertion("y", ONE),)),
    Block("Haux_T", Fraction(0), (Gradient(S), OneSidedB(Fraction(-1, 8)))),
    Block("Haux_u", Fraction(-3), (Insertion("g", S * U),)),
    Block("Haux_Y", Fraction(-3, 2), (Insertion("y", S), Insertion("v", -ONE / 8))),
    *(Block(f"R{2 * k}", Fraction(-k), (Insertion("g", S**k),)) for k in range(1, 5)),  # <R^2k> = <s^k>
]}
OBSERVABLES = {o.name: o for o in [
    Observable("M", (BLOCKS["M"],)),  # squared norm
    Observable("H", (BLOCKS["T"], BLOCKS["u"], BLOCKS["Y"])),  # H = -Delta/2 + u + Y
    Observable("Haux", (BLOCKS["Haux_T"], BLOCKS["Haux_u"], BLOCKS["Haux_Y"])),  # (1/16) sum_alpha Q_alpha R^2 Q_alpha
    *(Observable(k, (BLOCKS[k],)) for k in ("T", "u", "Y")),  # the three terms of H
    *(Observable(f"R{2 * k}", (BLOCKS[f"R{2 * k}"],)) for k in range(1, 5)),
]}


@dataclass
class Group:
    """Trial pairs whose cores lie in classes (A, B) with degree sum D.

    Rows are the core pairs (cj, ch), columns the pairs (k, l) of type indices of
    types(A) x types(B); ``j``, ``h`` give the trial positions of each (row, column) entry.
    """

    degree: int
    pairs: list
    type_pairs: list
    j: np.ndarray
    h: np.ndarray


def longitudinal_overlap(F: Scalars, b, d):
    """sqrt(2 sqrt(b d) / (b + d)), the overlap of normalized one-dimensional Gaussians (1 if b = d).

    The masters omit pi^nu beta^{-1/2}, beta = (b + d)/2, so with the reduced trial wavefunctions
    tilde psi_j = b_j^{1/4} pi^{-nu/2} f_j P_j, each with its own longitudinal width,
    a matrix element is this overlap times the assembled master sum: unequal
    widths need ball arithmetic, and the factor is invariant under dilations.
    """
    return F(1) if b == d else F.sqrt(2 * F.sqrt(F(b * d)) / F(b + d))


class Assembler:
    """Reduced block matrices of a list of trials, assembled by dense products in a number system.

    A trial (a, b, exponents, core) has the type (a, b, exponents); cores used by the same
    set of types form a class.  For classes A, B and a degree sum D the core pairs of A x B
    with d_cj + d_ch = D share the polynomial basis of every family, and each of them occurs
    with every type pair of types(A) x types(B).  A family component (family, transposed, b)
    therefore contributes phi @ SM to all these trial pairs at once: phi (core pairs x basis)
    holds the family coefficients, SM (basis x type pairs) the masters contracted with the
    structure polynomials, sum_rho q_rho master(mu + rho, atom), of each type pair.
    """

    def __init__(self, trials, table, norm=Fraction(1), scalars: Scalars = RATIONALS):
        self.table, self.norm, self.scalars = table, Fraction(norm), scalars
        self.types = sorted({t[:3] for t in trials}, key=str)
        self.data = [TrialData(*ty) for ty in self.types]
        kind = {ty: k for k, ty in enumerate(self.types)}
        keys = [(kind[t[:3]], t[3]) for t in trials]
        distinct = list(dict.fromkeys(keys))  # repeated trials share their rows and columns
        position = {key: i for i, key in enumerate(distinct)}
        self.size, self.expand = len(distinct), np.array([position[key] for key in keys])
        users = defaultdict(list)
        for k, core in distinct:
            users[core].append(k)
        classes = defaultdict(list)
        for core, ks in sorted(users.items()):
            classes[tuple(sorted(ks))].append(core)
        self.groups = []
        for (types_a, cores_a), (types_b, cores_b) in product(classes.items(), repeat=2):
            type_pairs = list(product(types_a, types_b))
            by_degree = defaultdict(list)
            for cj, ch in product(cores_a, cores_b):
                by_degree[table.degrees[cj] + table.degrees[ch]].append((cj, ch))
            for D, pairs in sorted(by_degree.items()):
                j = np.array([[position[k, cj] for k, _ in type_pairs] for cj, _ in pairs])
                h = np.array([[position[l, ch] for _, l in type_pairs] for _, ch in pairs])
                self.groups.append(Group(D, pairs, type_pairs, j, h))
        widths = {tj[:2] + th[:2] for tj in self.types for th in self.types}
        integrals = {w: OrientationIntegrals(*w, scalars) for w in widths}
        self.orientation = [[integrals[tj[:2] + th[:2]] for th in self.types] for tj in self.types]
        with scalars.precision():
            self.overlap = [[longitudinal_overlap(scalars, tj[1], th[1]) for th in self.types] for tj in self.types]
        self._coefficients, self._lookup = {}, {}

    def coefficients(self, component, g: int):
        """(phi, basis) of a family component on group g, or None where the family vanishes."""
        key = (component, g)
        if key not in self._coefficients:
            fam, transposed, b = component
            diagonal, offset, parity = FAMILY_SHAPE[fam]
            group, ft = self.groups[g], self.table.families[fam]
            self._coefficients[key] = None
            if group.degree % 2 == parity and group.degree + offset >= 0:
                N = (group.degree + offset) // 2
                if fam not in self._lookup:
                    self._lookup[fam] = ft.lookup()
                rows = [self._lookup[fam][(ch, cj) if transposed else (cj, ch)][1] for cj, ch in group.pairs]
                basis = ft.basis(N)
                cols = [i for i, m in enumerate(basis) if not diagonal or m[0] == b]
                if cols:
                    integers = ft.numerators[N][np.ix_(rows, cols)].tolist()
                    self._coefficients[key] = (self.scalars.integer_matrix(integers, ft.denominators[N]),
                                               [basis[c][1] if diagonal else basis[c] for c in cols])
        return self._coefficients[key]

    def matrix(self, name):
        """The reduced block matrix (fmpq_mat or arb_mat, by the number system)."""
        F = self.scalars
        with F.precision():
            terms = {}  # (k, l) -> component -> [(atom, [(rho, q_rho)])]
            for (k, tj), (l, th) in product(enumerate(self.data), repeat=2):
                terms[k, l] = defaultdict(list)
                for fam, transposed, b, atom, q in BLOCKS[name].terms(tj, th):
                    terms[k, l][fam, transposed, b].append((atom, [(r, F(c)) for r, c in q.terms.items()]))
            components = {c for t in terms.values() for c in t}
            columns = {}  # (component, degree, k, l) -> contracted masters on the basis
            out = np.full((self.size, self.size), F(0), dtype=object)
            norm = F(self.norm)
            for g, group in enumerate(self.groups):
                total = None
                for component in components:
                    found = self.coefficients(component, g)
                    if found is None:
                        continue
                    phi, basis = found
                    for k, l in group.type_pairs:
                        if (component, group.degree, k, l) not in columns:
                            O, parts = self.orientation[k][l], terms[k, l].get(component, ())
                            columns[component, group.degree, k, l] = [
                                self.overlap[k][l] * sum((c * O.master((mu[0] + r[0], mu[1] + r[1], mu[2] + r[2]), atom)
                                                          for atom, q in parts for r, c in q), F(0)) for mu in basis]
                    SM = F.matrix([list(row) for row in zip(*(columns[component, group.degree, k, l]
                                                               for k, l in group.type_pairs))])
                    total = phi * SM if total is None else total + phi * SM
                if total is not None:
                    out[group.j, group.h] = np.array((total * norm).tolist(), dtype=object)
            return F.matrix(out[np.ix_(self.expand, self.expand)].tolist())
