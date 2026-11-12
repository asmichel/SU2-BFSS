"""Polynomial singlets (covariants), defined by data and evaluated on the gauge slice.

A covariant P(X) is a Fock-valued polynomial with P(O X V^T) = U(O, V) P(X) (the paper's Eq. (9)).
It is determined by its restriction to the gauge slice
D(x) = diag(x_1, x_2, x_3) (the paper, Sec. III A), where it is a polynomial in x with
values in the invariant subspace F_*.  Every seed is therefore specified by
its restriction to the slice, either directly (tensor data or a slice formula in
x, y, z) or through an equivariant operator expression such as

    (126*I - C2)*Y(X)^2*S - 3360*s*S

over the model vocabulary below.  Expressions are evaluated right to left on
polynomial vectors, so operator products are never formed explicitly.

Vocabulary: scalars s, u, v (invariants), x, y, z (the singular values x_1, x_2, x_3, only in
slice formulas), sqrt(2), division by rational constants; operators I, N (fermion number), C2 (colour
Casimir), Ksp (spatial Casimir), Y(M) = sum M_ai K_ai, Pi_j(A) (Lagrange projector
onto the eigenvalue j(j+1) of A among 0, 2, 6, 12), [A, B]; matrices X, W(M) =
(tr(M M^T) - M M^T) M, YM(M)_dc = sum_i M_di K_ci, with tr(.) and transpose ^T.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import prod

from .algebra import Polynomial
from .fock import Q8, SQRT2


# ---------------------------------------------------------------------------
# Values on the gauge slice
# ---------------------------------------------------------------------------
class Poly(Polynomial):
    """Scalar polynomial in the slice coordinates x_1, x_2, x_3 with Q(zeta_8) coefficients."""

    coerce = staticmethod(Q8.coerce)

    @property
    def grade(self):
        grades = {c.grade for c in self.terms.values()}
        assert len(grades) <= 1
        return grades.pop() if grades else 0


S_INV = sum((Poly.variable(a) ** 2 for a in range(3)), Poly())
U_INV = sum((Poly.variable(a) ** 2 * Poly.variable(b) ** 2 for a in range(3) for b in range(a + 1, 3)), Poly())
V_INV = Poly.variable(0) ** 2 * Poly.variable(1) ** 2 * Poly.variable(2) ** 2


@dataclass
class PolyVec:
    """Fock-valued polynomial on the slice: monomial -> coefficient vector, with a reality grade."""

    terms: dict
    grade: int = 0

    def __add__(self, o: PolyVec):
        if not o.terms:
            return self
        if not self.terms:
            return o
        assert self.grade == o.grade, "adding real and imaginary vectors"
        t = dict(self.terms)
        for m, v in o.terms.items():
            t[m] = t[m] + v if m in t else v
        return PolyVec(t, self.grade)


class Operator(ABC):
    """Operator field on the slice, acting on polynomial vectors."""

    grade = 0

    @abstractmethod
    def apply(self, vec: PolyVec, ctx) -> PolyVec: ...

    def __mul__(self, o):
        return Product([self, o])

    def __add__(self, o):
        return Sum([self, o])

    def __neg__(self):
        return Product([ScalarOp(Poly.const(-1)), self])

    def __sub__(self, o):
        return Sum([self, -o])


class ScalarOp(Operator):
    def __init__(self, poly: Poly):
        self.poly, self.grade = poly, poly.grade

    def apply(self, vec, ctx):
        out = defaultdict(lambda: 0)
        for mp, c in self.poly.terms.items():
            r = ctx.scalar(c)
            for mv, v in vec.terms.items():
                m = tuple(a + b for a, b in zip(mp, mv))
                out[m] = ctx.reduce(out[m] + r * v)
        return PolyVec(dict(out), (vec.grade + self.grade) % 2)


class BaseOp(Operator):
    def __init__(self, name: str, grade: int):
        self.name, self.grade = name, grade

    def apply(self, vec, ctx):
        return PolyVec({m: ctx.act(self.name, v) for m, v in vec.terms.items()}, (vec.grade + self.grade) % 2)


class Product(Operator):
    def __init__(self, factors):
        self.factors = factors
        self.grade = sum(f.grade for f in factors) % 2

    def apply(self, vec, ctx):
        for f in reversed(self.factors):
            vec = f.apply(vec, ctx)
        return vec


class Sum(Operator):
    def __init__(self, terms):
        self.terms = terms
        grades = {t.grade for t in terms}
        assert len(grades) == 1, "sum of real and imaginary operators"
        self.grade = grades.pop()

    def apply(self, vec, ctx):
        return reduce(lambda a, b: a + b, (t.apply(vec, ctx) for t in self.terms))


IDENTITY = ScalarOp(Poly.const(1))


def as_operator(x) -> Operator:
    return x if isinstance(x, Operator) else ScalarOp(x if isinstance(x, Poly) else Poly.const(x))


@dataclass
class DiagMatrix:
    """3x9 matrix field at D(x): only the leading diagonal entries are nonzero."""

    d: list


@dataclass
class OpMatrix:
    """3x3 matrix of operators (rows d, columns c)."""

    e: list

    def __mul__(self, o: OpMatrix):
        return OpMatrix([[Sum([self.e[a][b] * o.e[b][c] for b in range(3)]) for c in range(3)] for a in range(3)])

    def T(self):
        return OpMatrix([[self.e[c][d] for c in range(3)] for d in range(3)])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z_0-9]*)|(.))")


def tokenize(src: str):
    out = []
    for num, name, sym in TOKEN.findall(src):
        out.append(("num", int(num)) if num else ("name", name) if name else ("sym", sym))
    return [t for t in out if t != ("sym", "")]


class Parser:
    def __init__(self, src):
        self.toks, self.i = tokenize(src), 0

    def peek(self, *alts):
        return self.i < len(self.toks) and self.toks[self.i][1] in alts

    def take(self, expect=None):
        tok = self.toks[self.i]
        assert expect is None or tok[1] == expect, f"expected {expect}, got {tok}"
        self.i += 1
        return tok

    def parse(self):
        e = self.expr()
        assert self.i == len(self.toks), f"trailing input at {self.toks[self.i:]}"
        return e

    def expr(self):
        e = self.term()
        while self.peek("+", "-"):
            e = (self.take()[1], e, self.term())
        return e

    def term(self):
        e = self.unary()
        while self.peek("*", "/"):
            e = (self.take()[1], e, self.unary())
        return e

    def unary(self):
        if self.peek("-"):
            self.take()
            return ("neg", self.unary())
        return self.power()

    def power(self):
        e = self.atom()
        while self.peek("^"):
            self.take()
            e = ("T", e) if self.peek("T") and self.take() else ("^", e, self.take()[1])
        return e

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return ("const", val)
        if val == "(":
            e = self.expr()
            self.take(")")
            return e
        if val == "[":
            a = self.expr()
            self.take(",")
            b = self.expr()
            self.take("]")
            return ("comm", a, b)
        if self.peek("("):
            self.take("(")
            args = [self.expr()]
            while self.peek(","):
                self.take()
                args.append(self.expr())
            self.take(")")
            return ("call", val, args)
        return ("name", val)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CASIMIR_SPINS = (0, 1, 2, 3)


class Evaluator:
    """Evaluates expressions to scalars, matrices, operators or polynomial vectors."""

    SCALARS = {"s": S_INV, "u": U_INV, "v": V_INV, "x": Poly.variable(0), "y": Poly.variable(1), "z": Poly.variable(2)}

    def __init__(self, ctx, environment):
        self.ctx, self.env = ctx, environment

    def __call__(self, src: str):
        return self.eval(Parser(src).parse())

    def eval(self, node):
        kind = node[0]
        if kind == "const":
            return Poly.const(node[1])
        if kind == "name":
            return self.name(node[1])
        if kind == "neg":
            v = self.eval(node[1])
            return self.mul(Poly.const(-1), v)
        if kind in "+-":
            a, b = self.eval(node[1]), self.eval(node[2])
            if kind == "-":
                b = self.mul(Poly.const(-1), b)
            return self.add(a, b)
        if kind == "*":
            return self.mul(self.eval(node[1]), self.eval(node[2]))
        if kind == "/":
            d = self.eval(node[2])
            assert isinstance(d, Poly) and set(d.terms) == {(0, 0, 0)}, "division only by constants"
            c = d.terms[(0, 0, 0)].c
            if any(c[1:]):
                raise ValueError(f"division by the irrational constant {d.terms[(0, 0, 0)]}: multiply instead")
            return self.mul(Poly.const(1 / c[0]), self.eval(node[1]))
        if kind == "^":
            base = self.eval(node[1])
            return reduce(self.mul, [base] * node[2]) if node[2] else Poly.const(1)
        if kind == "T":
            return self.eval(node[1]).T()
        if kind == "comm":
            a, b = as_operator(self.eval(node[1])), as_operator(self.eval(node[2]))
            return a * b - b * a
        if kind == "call":
            return self.call(node[1], [self.eval(x) for x in node[2]])
        raise ValueError(node)

    def name(self, name):
        if name in self.SCALARS:
            return self.SCALARS[name]
        if name == "I":
            return IDENTITY
        if name == "X":
            return DiagMatrix([Poly.variable(a) for a in range(3)])
        if name in self.ctx.operators:
            return self.operator(name)
        return self.env(name)

    def operator(self, name):
        return BaseOp(name, self.ctx.grade(name))

    def call(self, f, args):
        if f == "sqrt":
            assert set(args[0].terms) == {(0, 0, 0)} and args[0].terms[(0, 0, 0)] == Q8.coerce(2)
            return Poly.const(SQRT2)
        if f == "Y":
            return Sum([ScalarOp(m) * self.operator(f"K{a}{a}") for a, m in enumerate(args[0].d)])
        if f == "YM":
            return OpMatrix([[ScalarOp(args[0].d[d]) * self.operator(f"K{c}{d}") for c in range(3)] for d in range(3)])
        if f == "W":
            m = args[0].d
            trace = sum((x * x for x in m), Poly())
            return DiagMatrix([(trace - x * x) * x for x in m])
        if f == "tr":
            return Sum([args[0].e[a][a] for a in range(3)])
        if f.startswith("Pi_"):
            j, A = int(f[3:]), as_operator(args[0])
            lam = lambda k: k * (k + 1)
            factors = [(A - as_operator(Poly.const(lam(k)))) for k in CASIMIR_SPINS if k != j]
            norm = prod(Fraction(lam(j) - lam(k)) for k in CASIMIR_SPINS if k != j)
            return ScalarOp(Poly.const(1 / norm)) * Product(factors)
        raise ValueError(f"unknown function {f}")

    def add(self, a, b):
        if isinstance(a, Poly) and isinstance(b, Poly):
            return a + b
        if isinstance(a, PolyVec):
            return a + b
        if isinstance(a, OpMatrix):
            return OpMatrix([[a.e[i][j] + b.e[i][j] for j in range(3)] for i in range(3)])
        return as_operator(a) + as_operator(b)

    def mul(self, a, b):
        if isinstance(b, PolyVec):
            return as_operator(a).apply(b, self.ctx)
        if isinstance(a, Poly) and isinstance(b, Poly):
            return a * b
        if isinstance(a, OpMatrix) and isinstance(b, OpMatrix):
            return a * b
        if isinstance(a, Poly) and isinstance(b, OpMatrix):
            return OpMatrix([[ScalarOp(a) * x for x in row] for row in b.e])
        if isinstance(a, Poly) and isinstance(b, DiagMatrix):
            return DiagMatrix([a * x for x in b.d])
        return as_operator(a) * as_operator(b)


# ---------------------------------------------------------------------------
# Polynomial singlets
# ---------------------------------------------------------------------------
class PolynomialSinglet(ABC):
    """Polynomial singlet (the paper, Sec. III B): an equivariant Fock-valued polynomial, homogeneous of a given
    Cartesian degree."""

    name: str
    degree: int

    @abstractmethod
    def on_slice(self, ctx, env) -> PolyVec:
        """Restriction to the gauge slice, as a polynomial vector in the context's space."""


class Seed(PolynomialSinglet):
    """Named generator of the covariant module, defined by data (tensors or an expression)."""

    @abstractmethod
    def to_json(self) -> dict: ...


@dataclass
class TensorSeed(Seed):
    """Seed given by orbit-tensor coefficients of monomials in the slice coordinates."""

    name: str
    degree: int
    tensors: list  # list of (monomial, rows) with rows [[sorted flavour masks], scalar]
    denominator: int

    def on_slice(self, ctx, env):
        return PolyVec({tuple(m): ctx.orbit_vector(rows, self.denominator) for m, rows in self.tensors}, 0)

    def to_json(self):
        return {"name": self.name, "degree": self.degree, "kind": "tensor", "denominator": str(self.denominator),
                "monomials": [list(m) for m, _ in self.tensors], "tensors": [rows for _, rows in self.tensors]}


@dataclass
class ExpressionSeed(Seed):
    """Seed given by an operator expression applied to other seeds, or by a formula on the gauge slice
    (``slice_formula``; kind "canonical" in the JSON form, whose digest keys the cached contraction tables)."""

    name: str
    degree: int
    expression: str
    slice_formula: bool = False

    def on_slice(self, ctx, env):
        value = Evaluator(ctx, env)(self.expression)
        assert isinstance(value, PolyVec), f"{self.name} does not evaluate to a vector"
        assert all(sum(m) == self.degree for m in value.terms), f"{self.name} is not homogeneous of degree {self.degree}"
        return value

    def to_json(self):
        return {"name": self.name, "degree": self.degree, "kind": "canonical" if self.slice_formula else "expression",
                "definition": self.expression}


class OperatorImage(PolynomialSinglet):
    """O(X) P for an equivariant operator field O of fixed degree: maps covariants to covariants."""

    base: PolynomialSinglet

    @property
    @abstractmethod
    def label(self) -> str: ...

    @property
    @abstractmethod
    def shift(self) -> int:
        """Degree of the operator field."""

    @abstractmethod
    def operator(self, ctx, env) -> Operator: ...

    @property
    def name(self):
        return f"{self.label} {self.base.name}"

    @property
    def degree(self):
        return self.base.degree + self.shift

    def on_slice(self, ctx, env):
        return self.operator(ctx, env).apply(self.base.on_slice(ctx, env), ctx)


@dataclass
class YukawaPower(OperatorImage):
    """Y(X)^p P: the Yukawa field is an equivariant operator field."""

    base: PolynomialSinglet
    power: int

    @property
    def label(self):
        return f"Y^{self.power}"

    @property
    def shift(self):
        return self.power

    def operator(self, ctx, env):
        return Product([Evaluator(ctx, env)("Y(X)")] * self.power)


@dataclass
class InvariantMultiple(OperatorImage):
    """s^k u^l v^m P: covariants form a module over the invariant polynomials."""

    exponents: tuple
    base: PolynomialSinglet

    @property
    def label(self):
        return "s^{} u^{} v^{}".format(*self.exponents)

    @property
    def shift(self):
        k, l, m = self.exponents
        return 2 * k + 4 * l + 6 * m

    def operator(self, ctx, env):
        k, l, m = self.exponents
        return ScalarOp(S_INV**k * U_INV**l * V_INV**m)


class SeedSet:
    """Ordered seeds plus named operator fields; resolves references between definitions."""

    def environment(self, ctx):
        """Name resolver for expressions: fields and seeds, evaluated once per context."""
        cache = {}

        def env(name):
            if name not in cache:
                if name in self.fields:
                    cache[name] = Evaluator(ctx, env)(self.fields[name])
                elif name in self.by_name:
                    cache[name] = self.by_name[name].on_slice(ctx, env)
                else:
                    raise KeyError(f"undefined name {name}")
            return cache[name]

        return env

    def __init__(self, seeds: list[PolynomialSinglet], fields: dict[str, str] | None = None, norm=Fraction(1)):
        # ``norm`` is the factor multiplying every inner product of seed-built vectors: tensor
        # data carry one common irrational scale (1/sqrt(1144) for the published seeds).
        self.seeds, self.fields, self.norm = seeds, dict(fields or {}), Fraction(norm)
        self.by_name = {s.name: s for s in seeds}
        self.index = {s.name: i for i, s in enumerate(seeds)}

    def __len__(self):
        return len(self.seeds)

    def evaluate(self, ctx) -> list[PolyVec]:
        """Restrictions of all seeds to the gauge slice, in the given context."""
        env = self.environment(ctx)
        return [env(s.name) for s in self.seeds]

    def to_json(self):
        return {"norm": str(self.norm), "fields": self.fields, "seeds": [s.to_json() for s in self.seeds]}

    @classmethod
    def from_json(cls, data):
        seeds = []
        for s in data["seeds"]:
            if s["kind"] == "tensor":
                seeds.append(TensorSeed(s["name"], s["degree"], list(zip(map(tuple, s["monomials"]), s["tensors"])),
                                        int(s["denominator"])))
            else:
                seeds.append(ExpressionSeed(s["name"], s["degree"], s["definition"], s["kind"] == "canonical"))
        return cls(seeds, data.get("fields"), Fraction(data.get("norm", 1)))
