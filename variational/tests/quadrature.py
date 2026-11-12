"""Independent numerical oracle: integrals over singular values of gauge-slice integrands.

For an invariant integrand F(X) = F(D(x)), int d^27X F = pi^{27/2} I[F] / I[e^{-s}] with
I[h] = int_{x1>x2>x3>0} Delta(x^2) (x1 x2 x3)^6 h dx.  Profiles and their derivatives on the
slice come from quadrature over the sphere octant: f = <e^{...}>, d f / d x_a = x_a (-a f + 2 eta P1_a).
"""

from __future__ import annotations

import numpy as np

from bfss.algebra import invariants


def sphere_octant(n=40):
    c, wc = np.polynomial.legendre.leggauss(n)
    c, wc = (c + 1) / 2, wc / 2
    ph, wp = np.polynomial.legendre.leggauss(n)
    ph, wp = (ph + 1) * np.pi / 4, wp * np.pi / 4
    C, P = np.meshgrid(c, ph, indexing="ij")
    n2 = np.stack([(1 - C**2) * np.cos(P) ** 2, (1 - C**2) * np.sin(P) ** 2, C**2]).reshape(3, -1)
    return n2, (wc[:, None] * wp[None, :]).ravel() * 8 / (4 * np.pi)


class SliceQuadrature:
    """Gauss-Legendre grid over ordered singular values with profile data for given widths."""

    def __init__(self, n=40, L=6.5):
        t, w = np.polynomial.legendre.leggauss(n)
        t, w = (t + 1) * L / 2, w * L / 2
        T = np.stack(np.meshgrid(t, t, t, indexing="ij")).reshape(3, -1)
        W = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
        x3 = T[2]
        x2 = x3 + T[1]
        self.x = np.stack([x2 + T[0], x2, x3])
        self.g = self.x**2
        self.e = invariants(self.g)  # s, u, v
        g = self.g
        self.jac = W * (g[0] - g[1]) * (g[0] - g[2]) * (g[1] - g[2]) * self.x.prod(0) ** 6
        self.norm = np.sum(self.jac * np.exp(-g.sum(0)))
        self.n2, self.wn = sphere_octant()

    def moments(self, a, b):
        """f_{a,b}(D(x)) and P1_a = (1/4 pi) int n_a^2 exp(-a s/2 + eta n^T G n) dOmega on the grid."""
        eta = (a - b) / 2
        f, P1 = np.empty(self.g.shape[1]), np.empty(self.g.shape)
        for sl in np.array_split(np.arange(self.g.shape[1]), 32):
            gs = self.g[:, sl]
            E = np.exp(eta * (self.n2.T @ gs) - a * gs.sum(0) / 2) * self.wn[:, None]
            f[sl], P1[:, sl] = E.sum(0), self.n2 @ E
        return f, P1

    def profile(self, a, b):
        """(f, grad f) on the grid: f_{a,b}(D(x)) and d f / d x_a = x_a (-a f + 2 eta P1_a)."""
        f, P1 = self.moments(a, b)
        return f, self.x * (-a * f + (a - b) * P1)

    def integrate(self, values, beta):
        """Reduced integral: divide pi^nu beta^{-1/2} out of int d^27X F."""
        return np.sqrt(beta) * np.sum(self.jac * values) / self.norm
