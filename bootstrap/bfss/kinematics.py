"""Kinematics: the Weyl-Clifford algebra of SU(2) BFSS.

27 bosonic modes with [x_A, p_B] = i delta_AB and 48 Majorana fermions with
{theta_i, theta_j} = delta_ij.  A basis monomial is a tuple (xs, ps, mask): the
Weyl-ordered product of x^xs p^ps (sorted mode tuples, repetitions allowed) times
i^{k(k-1)/2} theta_{j1}...theta_{jk} for the set bits j1 < ... < jk of mask.  Every
basis monomial is Hermitian, so its expectation value is real.

Coefficients lie in Q(i, sqrt2).  A polynomial stores integer 4-tuples (a, b, c, d)
meaning (a + b sqrt2 + i (c + d sqrt2)) / den, with one positive denominator per
polynomial.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
from math import factorial, gcd

IDENTITY = ((), (), 0)


def boson(a: int, I: int) -> int:
    return 9 * a + I


def fermion(a: int, alpha: int) -> int:
    return 16 * a + alpha


# ---- numbers in Q(i, sqrt2): integer 4-tuples (a, b, c, d) = a + b sqrt2 + i (c + d sqrt2)
ONE, I, SQRT2 = (1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0)


def qmul(u, v):
    a, b, c, d = u
    e, f, g, h = v
    return (a * e + 2 * b * f - c * g - 2 * d * h,
            a * f + b * e - c * h - d * g,
            a * g + c * e + 2 * b * h + 2 * d * f,
            a * h + d * e + b * g + c * f)


def qrot(u, k: int):
    """Multiply by i^k."""
    a, b, c, d = u
    k %= 4
    if k == 0:
        return u
    if k == 1:
        return (-c, -d, a, b)
    if k == 2:
        return (-a, -b, -c, -d)
    return (c, d, -a, -b)


class Poly:
    """Sparse polynomial: terms {monomial: (a, b, c, d)} over the positive integer den."""

    __slots__ = ("terms", "den")

    def __init__(self, terms=None, den: int = 1):
        self.terms = terms if terms is not None else {}
        self.den = den

    @classmethod
    def mono(cls, m, coef=ONE, den: int = 1) -> "Poly":
        return cls({m: coef}, den).normal()

    def normal(self) -> "Poly":
        """Drop zero terms and divide numerators and denominator by their gcd."""
        self.terms = {m: c for m, c in self.terms.items() if any(c)}
        g = self.den
        for c in self.terms.values():
            for v in c:
                g = gcd(g, v)
                if g == 1:
                    return self
        if g > 1:
            self.den //= g
            self.terms = {m: tuple(v // g for v in c) for m, c in self.terms.items()}
        return self

    def __eq__(self, other):
        return isinstance(other, Poly) and self.den == other.den and self.terms == other.terms

    def __add__(self, other: "Poly") -> "Poly":
        den = self.den * other.den // gcd(self.den, other.den)
        s, t = den // self.den, den // other.den
        acc = {m: tuple(s * v for v in c) for m, c in self.terms.items()}
        for m, c in other.terms.items():
            old = acc.get(m)
            acc[m] = tuple(t * v for v in c) if old is None else tuple(x + t * v for x, v in zip(old, c))
        return Poly(acc, den).normal()

    def __neg__(self) -> "Poly":
        return Poly({m: tuple(-v for v in c) for m, c in self.terms.items()}, self.den)

    def __sub__(self, other: "Poly") -> "Poly":
        return self + (-other)

    def scale(self, coef=ONE, num: int = 1, den: int = 1) -> "Poly":
        """Multiply by coef * num / den (coef in Q(i, sqrt2), num and den integers)."""
        terms = {m: tuple(num * v for v in qmul(coef, c)) for m, c in self.terms.items()}
        return Poly(terms, self.den * den).normal()

    def dagger(self) -> "Poly":
        return Poly({m: (a, b, -c, -d) for m, (a, b, c, d) in self.terms.items()}, self.den)


def psum(polys) -> Poly:
    """Sum of an iterable of polynomials."""
    out = Poly()
    for p in polys:
        out = out + p
    return out


# ---- products
def falling(n: int, k: int) -> int:
    """Falling factorial n (n-1) ... (n-k+1)."""
    return factorial(n) // factorial(n - k)


@lru_cache(maxsize=None)
def _mode_options(lx: int, lp: int, rx: int, rp: int):
    """One-mode Moyal product x^lx p^lp * x^rx p^rp = sum num i^c / 2^c x^.. p^.. (c contractions)."""
    out = defaultdict(int)
    for k in range(min(lx, rp) + 1):          # left x with right p: factor (i/2)^k / k!
        for l in range(min(lp, rx) + 1):      # left p with right x: factor (-i/2)^l / l!
            num = falling(lx, k) * falling(rp, k) * falling(lp, l) * falling(rx, l)
            out[(lx + rx - k - l, lp + rp - k - l, k + l)] += (-1) ** l * num // (factorial(k) * factorial(l))
    return tuple((x, p, n, c) for (x, p, c), n in sorted(out.items()) if n)


@lru_cache(maxsize=1 << 21)
def _boson_options(lxs, lps, rxs, rps):
    """Moyal product of two bosonic words: tuple of (xs, ps, numerator, contractions)."""
    lx, lp, rx, rp = Counter(lxs), Counter(lps), Counter(rxs), Counter(rps)
    modes = sorted(set(lx) | set(lp) | set(rx) | set(rp))
    per_mode = [_mode_options(lx[m], lp[m], rx[m], rp[m]) for m in modes]
    out = []
    for choice in product(*per_mode):
        xs, ps, num, c = [], [], 1, 0
        for m, (x, p, n, k) in zip(modes, choice):
            xs += [m] * x
            ps += [m] * p
            num *= n
            c += k
        out.append((tuple(xs), tuple(ps), num, c))
    return tuple(out)


@lru_cache(maxsize=1 << 21)
def _fermion_product(L: int, R: int):
    """Hermitian Clifford words: L * R = i^phase sign 2^-overlap (L xor R)."""
    inversions, scan = 0, R
    while scan:
        low = scan & -scan
        inversions += (L >> low.bit_length()).bit_count()
        scan ^= low
    S = L ^ R
    nl, nr, ns = L.bit_count(), R.bit_count(), S.bit_count()
    phase = (nl * (nl - 1) + nr * (nr - 1) - ns * (ns - 1)) // 2
    return S, -1 if inversions & 1 else 1, phase % 4, (L & R).bit_count()


def contract(lefts, rights, dagger_left: bool = True) -> Poly:
    """sum_c op(lefts[c]) * rights[c], op = dagger or identity; exact integer accumulation."""
    lefts, rights = list(lefts), list(rights)
    if len(lefts) != len(rights):
        raise ValueError("contracted operators need the same number of components")
    D = E = 1
    mlx = mlp = mlf = mrx = mrp = mrf = 0
    for P in lefts:
        D = D * P.den // gcd(D, P.den)
        for xs, ps, f in P.terms:
            mlx, mlp, mlf = max(mlx, len(xs)), max(mlp, len(ps)), max(mlf, f.bit_count())
    for P in rights:
        E = E * P.den // gcd(E, P.den)
        for xs, ps, f in P.terms:
            mrx, mrp, mrf = max(mrx, len(xs)), max(mrp, len(ps)), max(mrf, f.bit_count())
    cmax = min(mlf, mrf) + min(mlx, mrp) + min(mlp, mrx)
    acc = defaultdict(lambda: [0, 0, 0, 0])
    for P, R in zip(lefts, rights):
        sl, sr = D // P.den, E // R.den
        rterms = [(m, tuple(sr * v for v in c)) for m, c in R.terms.items()]
        for (lxs, lps, lf), (a, b, c, d) in P.terms.items():
            lc = (sl * a, sl * b, -sl * c, -sl * d) if dagger_left else (sl * a, sl * b, sl * c, sl * d)
            for (rxs, rps, rf), rc in rterms:
                q = qmul(lc, rc)
                S, sign, fph, ov = _fermion_product(lf, rf)
                for xs, ps, num, k in _boson_options(lxs, lps, rxs, rps):
                    w = sign * num << (cmax - ov - k)
                    qa, qb, qc, qd = qrot(q, fph + k)
                    t = acc[(xs, ps, S)]
                    t[0] += w * qa
                    t[1] += w * qb
                    t[2] += w * qc
                    t[3] += w * qd
    return Poly({m: tuple(v) for m, v in acc.items()}, D * E << cmax).normal()


def star(A: Poly, B: Poly) -> Poly:
    return contract([A], [B], dagger_left=False)
