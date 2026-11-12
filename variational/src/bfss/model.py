"""The SU(2) BFSS model in its explicit fermion representation (the paper, Sec. II and App. A).

Everything is constructed from the stated conventions: spherical Fock modes
m = 6A + 2c + rho, the Cartesian colour polarization U, Majoranas with real spinor
order (0,q),(0,p),...,(7,p), and the real gamma matrices built from H_i and B_m.
"""

from __future__ import annotations

from functools import cached_property
from itertools import permutations

import numpy as np

from .fock import Q8, SQRT2, FermionOperator, I

COLORS, DIMS, SPINOR = 3, 9, 16
NU = DIMS * COLORS / 2  # 27/2


def mode(c: int, rho: int, A: int) -> int:
    return 6 * A + 2 * c + rho


def levi_civita(a, b, c) -> int:
    return {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.get((a, b, c), 0)


def _flavor_rhos():
    def E(A, B):
        m = np.zeros((4, 4), dtype=complex)
        m[A - 1, B - 1], m[B - 1, A - 1] = 1, -1
        return m

    return [E(1, 2) + E(3, 4), 1j * (E(1, 2) - E(3, 4)), E(1, 3) - E(2, 4),
            1j * (E(1, 3) + E(2, 4)), E(1, 4) + E(2, 3), 1j * (E(1, 4) - E(2, 3))]


H_SPIN = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[-1, 0], [0, 1]])]
EPS = np.array([[0, 1], [-1, 0]])
H_LINEAR = [np.kron(np.eye(4), h) for h in H_SPIN]
B_ANTILINEAR = [np.kron(r, EPS) for r in _flavor_rhos()]


def _realify(M: np.ndarray, antilinear: bool) -> np.ndarray:
    """Real 16x16 matrix of z -> M z (or M conj z) on z = (q + i p)/sqrt2, interleaved (q_r, p_r)."""
    G = np.zeros((SPINOR, SPINOR), dtype=int)
    s = -1 if antilinear else 1
    G[0::2, 0::2], G[1::2, 1::2] = M.real, s * M.real
    G[0::2, 1::2], G[1::2, 0::2] = -s * M.imag, M.imag
    return G


GAMMA = np.array([_realify(H, False) for H in H_LINEAR] + [_realify(B, True) for B in B_ANTILINEAR])

# Cartesian colour polarization f_a = sum_c U[a, c] f_c, U entries exact in Q(zeta_8).
_H = SQRT2 / 2
U_COLOR = [[_H, Q8(), -_H], [I * _H, Q8(), I * _H], [Q8(), Q8((1, 0, 0, 0)), Q8()]]


class Model:
    """Fermionic operators of the model, built lazily and exactly."""

    @cached_property
    def theta(self) -> list[list[FermionOperator]]:
        """Majoranas theta[a][alpha] (zero-based Cartesian colour and real spinor index)."""
        out = []
        for a in range(COLORS):
            row = []
            for r in range(8):
                A, rho = divmod(r, 2)
                f = FermionOperator({(2 * mode(c, rho, A),): U_COLOR[a][c] for c in range(3)})
                fd = FermionOperator({(2 * mode(c, rho, A) + 1,): U_COLOR[a][c].conjugate() for c in range(3)})
                row += [(f + fd) * (SQRT2 / 2), (f - fd) * (-I * SQRT2 / 2)]
            out.append(row)
        return out

    def bilinear(self, a: int, b: int, M) -> FermionOperator:
        """sum_{alpha,beta} M[alpha,beta] theta^a_alpha theta^b_beta for an integer 16x16 matrix M."""
        th = self.theta
        return sum((th[a][al] * th[b][be] * int(M[al, be]) for al, be in zip(*np.nonzero(M))), FermionOperator())

    def spinor(self, a: int, row) -> FermionOperator:
        """sum_beta row[beta] theta^a_beta for an integer row of a spinor matrix."""
        return sum((self.theta[a][be] * int(row[be]) for be in np.nonzero(row)[0]), FermionOperator())

    @cached_property
    def K(self) -> list[list[FermionOperator]]:
        """Yukawa operators K[a][i] = (i/sqrt2) sum eps_abc (Gamma_i)_{alpha beta} theta^b_alpha theta^c_beta."""
        return [[sum((self.bilinear(b, c, GAMMA[i]) * levi_civita(a, b, c) for b, c in permutations(range(3), 2)
                      if levi_civita(a, b, c)), FermionOperator()) * (I * SQRT2 / 2)
                 for i in range(DIMS)] for a in range(COLORS)]

    @cached_property
    def color_generators(self) -> dict[tuple[int, int], FermionOperator]:
        """Anti-Hermitian G_ab = sum_alpha theta^a_alpha theta^b_alpha for a < b."""
        return {(a, b): self.bilinear(a, b, np.eye(SPINOR, dtype=int)) for a in range(3) for b in range(a + 1, 3)}

    def spatial_generator(self, i: int, j: int) -> FermionOperator:
        """R_ij = (1/4) sum_a (Gamma_i Gamma_j)_{alpha beta} theta^a_alpha theta^a_beta."""
        GG = GAMMA[i] @ GAMMA[j]
        return sum((self.bilinear(a, a, GG) for a in range(COLORS)), FermionOperator()) / 4

    def C(self, alpha: int, a: int, i: int) -> FermionOperator:
        """C^alpha_{ai} = sum_beta (Gamma_i)_{alpha beta} theta^a_beta."""
        return self.spinor(a, GAMMA[i][alpha])

    @cached_property
    def number(self) -> FermionOperator:
        """N = sum_m f_m^dagger f_m over the 24 modes (4 flavours x 3 colours x 2 spins)."""
        return FermionOperator({(2 * m + 1, 2 * m): 1 for m in range(24)})

    @cached_property
    def color_casimir(self) -> FermionOperator:
        """C2 = -sum_{a<b} G_ab^2."""
        return -sum((g * g for g in self.color_generators.values()), FermionOperator())

    @cached_property
    def spatial_casimir(self) -> FermionOperator:
        """Ksp = -sum_{i<j} R_ij^2 over all 36 planes."""
        return -sum((R * R for i in range(DIMS) for j in range(i + 1, DIMS)
                     for R in [self.spatial_generator(i, j)]), FermionOperator())

    def V(self, alpha: int, X) -> FermionOperator:
        """Supercharge potential V_alpha(X) = sqrt2 sum_{i<j} eps_abc X_bi X_cj (Gamma_i Gamma_j)_{alpha beta} theta^a_beta."""
        out = FermionOperator()
        for i in range(DIMS):
            for j in range(i + 1, DIMS):
                row = (GAMMA[i] @ GAMMA[j])[alpha]
                for a in range(COLORS):
                    w = sum(levi_civita(a, b, c) * X[b][i] * X[c][j] for b in range(3) for c in range(3))
                    if w:
                        out += self.spinor(a, row) * w
        return out * SQRT2

    def c(self, alpha: int, X) -> FermionOperator:
        """c_alpha(X) = sum_{ai} X_ai C^alpha_{ai}, the supercharge contracted with the configuration."""
        return sum((self.C(alpha, a, i) * X[a][i] for a in range(COLORS) for i in range(DIMS) if X[a][i]), FermionOperator())

    def B(self, mu, nu) -> FermionOperator:
        """B_{ai,bk} = sum_alpha C^alpha_ai C^alpha_bk - 8 delta_ab delta_ik for the pairs mu = (a, i), nu = (b, k)."""
        (a, i), (b, k) = mu, nu
        return self.bilinear(a, b, GAMMA[i] @ GAMMA[k]) - 8 * int(mu == nu)

    def slice_operators(self) -> dict[str, FermionOperator]:
        """Named operators needed on the gauge slice (zero-based colour/space indices).

        K{a}{i}: Yukawa components with a, i < 3;  G{b}{k}, R{b}{k}: colour and spatial generators (the paper's
        G_bk and R_bk) in the plane b < k < 3;  RR{b} = sum_{k>=3} R_bk^2;  B{a}{b}{k} = B_{(a,a),(b,k)};
        BR{a}{b} = sum_{k>=3} B_{(a,a),(b,k)} R_bk;  cV{e}: coefficient of x^e in i sum_alpha c_alpha V_alpha.
        The names keep clear of the paper's S_ij (bosonic rotations), Q_alpha and V_alpha (Eq. (5)); the paper
        has no symbol for B_{ai,bk}, and its B (Eq. (24)) is a constant.
        """
        ops = {"N": self.number, "C2": self.color_casimir, "Ksp": self.spatial_casimir}
        ops.update({f"K{a}{i}": self.K[a][i] for a in range(3) for i in range(3)})
        R = {(b, k): self.spatial_generator(b, k) for b in range(3) for k in range(b + 1, DIMS)}
        for (b, k), g in self.color_generators.items():
            ops[f"G{b}{k}"], ops[f"R{b}{k}"] = g, R[b, k]
        for b in range(3):
            ops[f"RR{b}"] = sum((R[b, k] * R[b, k] for k in range(3, DIMS)), FermionOperator())
            for a in range(3):
                ops[f"BR{a}{b}"] = sum((self.B((a, a), (b, k)) * R[b, k] for k in range(3, DIMS)), FermionOperator())
                for k in range(3):
                    if (a, b, k) != (a, a, a):
                        ops[f"B{a}{b}{k}"] = self.B((a, a), (b, k))
        V = {}
        for a in range(3):
            for b, c in ((0, 1), (0, 2), (1, 2)):
                e = tuple(int(a == t) + int(b == t) + int(c == t) for t in range(3))
                term = sum((self.bilinear(a, a2, GAMMA[a] @ GAMMA[b] @ GAMMA[c]) * levi_civita(a2, b, c)
                            for a2 in range(3) if levi_civita(a2, b, c)), FermionOperator()) * (I * SQRT2)
                V[e] = V.get(e, FermionOperator()) + term
        ops.update({"cV" + "".join(map(str, e)): v for e, v in V.items() if len(v)})
        return ops


MODEL = Model()
