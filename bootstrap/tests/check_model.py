"""Associativity of the star product and Hermiticity of basis monomials on random inputs, and
the model identities, exactly.

  python tests/check_model.py
"""
import random
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the package root
from bfss import model as M
from bfss.kinematics import I, Poly, contract, psum, star


def comm(A, B):
    return star(A, B) - star(B, A)


def anti(A, B):
    return star(A, B) + star(B, A)


def check(name, lhs, rhs):
    ok = lhs == rhs
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    return ok


def random_monomial(rng):
    xs = tuple(sorted(rng.choices(range(27), k=rng.randint(0, 3))))
    ps = tuple(sorted(rng.choices(range(27), k=rng.randint(0, 2))))
    mask = sum(1 << b for b in rng.sample(range(48), rng.randint(0, 4)))
    return Poly.mono((xs, ps, mask))


def random_identities(trials: int = 300) -> bool:
    rng = random.Random(1)
    assoc = herm = 0
    for _ in range(trials):
        a, b, c = (random_monomial(rng) for _ in range(3))
        assoc += star(star(a, b), c) != star(a, star(b, c))
        herm += a.dagger() != a                     # basis monomials are Hermitian: real coefficients
    return check(f"associativity and Hermiticity ({trials} random triples)", (assoc, herm), (0, 0))


def main() -> None:
    t0 = time.time()
    H, Q, G, X = M.H(), M.Q(), M.gauss(), M.positions()
    good = [
        random_identities(),
        check("[X, P] = i", comm(M.X(0, 0), M.P(0, 0)), M.ONE.scale(I)),
        check("{theta, theta} = 1", anti(M.theta(1, 3), M.theta(1, 3)), M.ONE),
        check("[H, R2] = -2i D", comm(H, M.R2()), M.D().scale(I, num=-2)),
        check("[H, D] = i(-2T + 4u + Y)", comm(H, M.D()),
              (M.T().scale(num=-2) + M.u().scale(num=4) + M.Y()).scale(I)),
        check("[H, Y] = -i sum K P", comm(H, M.Y()), contract(M.K(), M.momenta(), dagger_left=False).scale(I, num=-1)),
        check("[H, gauss_a] = 0 (all 3)", [comm(H, g) for g in G], [Poly()] * 3),
        check("[H, J_IJ] = 0 (all 36)", [comm(H, j) for j in M.J_rot()], [Poly()] * 36),
        check("sum Q_a Q_a = 16 H", contract(Q, Q, dagger_left=False), H.scale(num=16)),
    ]
    # {Q_a, Q_b} = 2 delta H + 2 sum_{i,a} (Gamma_i)_ab X_ai gauss_a, all 136 pairs
    bad = 0
    for a, b in combinations_with_replacement(range(16), 2):
        rhs = psum(star(X[3 * J + c], G[c]).scale(num=2 * int(M.GAMMA[J][a, b]))
                   for J in range(9) for c in range(3) if M.GAMMA[J][a, b])
        if a == b:
            rhs = rhs + H.scale(num=2)
        bad += anti(Q[a], Q[b]) != rhs
    good.append(check(f"{{Q_a, Q_b}} = 2 delta H + 2 Gamma X gauss (136 pairs, {bad} wrong)", bad, 0))
    # Fierz: 720 - K.K = 8 R^+.R + 3/2 Psi.Psi
    psi = [psum(star(M.theta(l, al), M.theta(r, al)).scale(I, num=M.eps(c, l, r))
                for l in range(3) for r in range(3) if M.eps(c, l, r) for al in range(16)) for c in range(3)]
    good.append(check("Fierz 720 - K.K = 8 R^+.R + 3/2 Psi.Psi",
                      M.ONE.scale(num=720) - contract(M.K(), M.K()),
                      contract(M.R_rot(), M.R_rot()).scale(num=8) + contract(psi, psi).scale(num=3, den=2)))
    print(f"{sum(good)}/{len(good)} checks hold ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
