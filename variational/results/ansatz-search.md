# Pushing the fidelity bound: the ansatz search

Every number below is an optimization from scratch (`bfss optimize`, 128-bit balls) of the
published basis plus the listed shells. ⟨H_aux⟩ is the optimizer's smallest eigenvalue of the
weighted residual; measuring the saved states reproduces it to ~1e-11 (with rigorous radii of
order 1e-20), and F_lb = 1 - kappa ⟨H_aux⟩, kappa = (36 + 8 sqrt 14)/25. The target F_lb = 0.999 is
⟨H_aux⟩ = 3.79e-4. Widths (a, b) are those of the ansatz file: the published basis has (2, 1), (3, 1), (4, 1),
and the optimal dilation lambda sets the physical widths (lambda a, lambda b). `bfss optimize` divides the
widths of an ansatz by its largest b, so for bases with b = 2 or 3 the listed widths are not the canonical
ones, and lambda multiplies the divided widths (lambda a / b_max, lambda b / b_max).

Shell shorthands:

* **(a, b)**: the 224 bare cores Y^p B of the published basis (13 seed towers up to total degree
  19) at a new width;
* **(a, b) all**: the same with the invariant multiples the published basis has at its first
  width (729 functions);
* **s·hi**, **u·hi**: s or u times every bare core at the three published widths that lacks
  this multiple (the published basis has them only for the low Yukawa powers, so mostly p >= 9);
  **s²·pub**: s^2 times every bare core at the published widths;
* **inv(a, b)**: s, u and su times the bare cores at width (a, b).

## Starting point and diagnosis

The published 1951-function basis optimizes to ⟨H_aux⟩ = 3.4945e-3, F_lb = 0.99078 (lambda = 1.408).
The Coulomb-branch tests of the published state (`jaroslav/eft-tests` in the paper repository)
locate its shortcomings: every function carries the same longitudinal Gaussian, so the
r^-10 tail along the valley is cut off near r = 4.3, and the narrowest transverse profile matches
the W-boson ground state (width ~ r^-1/2) only up to r = 4.1. The tests put 34% of the energy E
beyond r = 4.3 and 58% below r = 3.8; ⟨H_aux⟩ weights the same local residual by R^2, which shifts the
balance outward but leaves the core significant.

## Round 1: one more width (224 functions each)

| added | N | lambda | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|
| (2, 1/2) | 2175 | 1.414 | 3.322e-3 | 0.99124 |
| (5, 1) | 2175 | 1.342 | 2.626e-3 | 0.99307 |
| (6, 1) | 2175 | 1.341 | 2.667e-3 | 0.99297 |
| (3, 1/2) | 2175 | 1.867 | 2.662e-3 | 0.99298 |
| (7, 1/4) | 2175 | 1.373 | 1.777e-3 | 0.99531 |
| (5, 1/4) | 2175 | 1.523 | 1.319e-3 | 0.99652 |
| (7, 1/2) | 2175 | 1.397 | 1.254e-3 | 0.99669 |
| **(5, 1/2)** | 2175 | 1.546 | **1.074e-3** | **0.99717** |
| (4, 1/2) with s, s^2 multiples | 2623 | 1.736 | 1.043e-3 | 0.99725 |
| s·hi | 2380 | 1.416 | 2.330e-3 | 0.99385 |
| u·hi | 2302 | 1.339 | 2.442e-3 | 0.99356 |
| s·hi, s²·pub | 3052 | 1.450 | 1.804e-3 | 0.99524 |

A broad longitudinal profile alone (small b) or a narrow transverse one alone (large a) removes
at most a quarter of ⟨H_aux⟩; together they extend the valley: at the optimal dilation a canonical
a resolves r ~ a, and b = 1/2 lets the profiles reach it. One such width removes 70% of ⟨H_aux⟩.

## Rounds 2 and 3: more valley widths, then the core

| basis | N | lambda | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|
| (5, 1/2) all | 2680 | 1.638 | 8.98e-4 | 0.99763 |
| (5, 1/2), (7, 1/4) | 2399 | 1.534 | 8.66e-4 | 0.99772 |
| (5, 1), (3, 1/2), (5, 1/2) | 2623 | 1.655 | 8.49e-4 | 0.99776 |
| (4, 1/2), (5, 1/2), (6, 1/2) | 2623 | 1.641 | 8.26e-4 | 0.99782 |
| (5, 1/2), (6, 1/2), (7, 1/4), (8, 1/4) | 2847 | 1.599 | 7.59e-4 | 0.99800 |
| (5, 1), (3, 1/2), (6, 1/2), (5, 1/4), (8, 1/4) | 3071 | 1.667 | 6.80e-4 | 0.99821 |
| **B6** = the last + (5, 1/2) | 3295 | 1.680 | **6.23e-4** | **0.99836** |
| B6 + (6, 1/8) | 3519 | 1.682 | 6.21e-4 | 0.99836 |
| B6 + (6, 1/4) | 3519 | 1.681 | 6.15e-4 | 0.99838 |
| B6 + (10, 1/8) | 3519 | 1.681 | 6.15e-4 | 0.99838 |
| B6 + (7, 1/2) | 3519 | 1.684 | 6.06e-4 | 0.99840 |
| B6 + (4, 1/2) | 3519 | 1.694 | 5.95e-4 | 0.99843 |
| B6 + inv(5, 1/2) | 3967 | 1.690 | 5.68e-4 | 0.99850 |
| B6 + s·hi | 3724 | 1.662 | 4.67e-4 | 0.99877 |
| **B6 + (3, 2)** | 3519 | 2.884 | **4.25e-4** | **0.99888** |

Further valley widths now gain at most 4.5%, while the core and the fermion content offer more.
A narrow longitudinal profile (b = 2; the canonical widths are
then halved, hence lambda ~ 2.9) and the s-multiples of the long Yukawa towers give 25-32%
each. The s-multiples let the relative weights of Y^p B change with the radius, which the
heavy-fermion projector on the valley requires: there Y^2/s tends to K_00^2.

## Round 4: core profiles

C1 = B6 + (3, 2) + s·hi.

| basis | N | lambda | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|
| B6 + (3, 2) + (4, 2) | 3743 | 2.917 | 4.01e-4 | 0.99894 |
| C1 | 3948 | 2.908 | 3.85e-4 | 0.99898 |
| C1 + (4, 2) | 4172 | 2.915 | 3.70e-4 | 0.99903 |
| C1 + (2, 2) | 4172 | 3.031 | 3.67e-4 | 0.99903 |
| C1 + s²·pub | 4620 | 2.958 | 3.53e-4 | 0.99907 |
| C1 + inv(3, 2) | 4620 | 2.985 | 3.34e-4 | 0.99912 |

## Round 5: on E0 = C1 + (4, 2) + (2, 2)

| basis | N | lambda | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|
| E0 | 4396 | 2.961 | 3.573e-4 | 0.99906 |
| E0 + Yukawa powers p+1..p+4 of every tower at the published widths | 4552 | 2.969 | 3.534e-4 | 0.99907 |
| E0 + powers p+1, p+2 at all widths | 4708 | 2.984 | 3.499e-4 | 0.99908 |
| E0 + s·(5, 1/2) | 4620 | 2.966 | 3.490e-4 | 0.99908 |
| E0 + s·(3, 2), s·(2, 2), s·(4, 2) | 5068 | 3.042 | 3.333e-4 | 0.99912 |

Longer Yukawa towers (up to total degree 23; the 276-core contraction table took 8:41 on 47
workers) add little: the towers are long enough once the profiles cover the valley. (Two runs
adding (4, 3) are not listed: on C1 the optimizer stopped at the edge of the old dilation
bracket (0.5, 4), since replaced by a wide grid; on E0 the isotropic width (2, 2), rescaled to
2/3, gave NaN matrix elements, fixed in the orientation integrals.)

## Round 6: combinations

F1 = E0 + inv(3, 2) + s²·pub. Gains of separate additions do not multiply: F1 removes 12.7%
of ⟨H_aux⟩ where the two additions alone removed 13.4% and 8.4% of C1's.

| basis | N | kept | lambda | ⟨H_aux⟩ | F_lb | time, memory |
|---|---|---|---|---|---|---|
| F1 | 5740 | 5736 | 3.100 | 3.120e-4 | 0.99918 | 39 min on 24 CPUs, 24 GB |
| F3 = F1 + s·(5, 1/2) + powers p+1, p+2 at all widths | 6276 | 6262 | 3.235 | 3.009e-4 | 0.99921 | 48 min on 24 CPUs, 31 GB |
| F2 = F1 + inv(2, 2) + inv(4, 2) | 7084 | 7071 | 3.049 | 2.946e-4 | 0.99922 | 81 min on 48 CPUs, 28 GB |
| G = F2 + F3's additions | 7620 | 7593 | 3.184 | 2.848e-4 | 0.99925 | 47 min on 40 CPUs, 44 GB |

(F2 ran with ball products for the frame transforms, G with the fixed-point products.)

## Saturation

Further structure on top of G lowers ⟨H_aux⟩ by 0.6-2.2% for 224-750 more functions:

| basis | N | kept | lambda | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|---|
| G + (5, 1/3), (7, 1/3) | 8068 | 8041 | 3.223 | 2.831e-4 | 0.99925 |
| G + (4, 3) | 7844 | 7817 | 4.723 | 2.831e-4 | 0.99925 |
| G + u·(5, 1/2), su·(5, 1/2) | 8120 | 8093 | 3.283 | 2.821e-4 | 0.99926 |
| G + (3, 3), (5, 3) | 8068 | 8039 | 4.706 | 2.814e-4 | 0.99926 |
| **G + v·(3, 2), v·(2, 2), v·(4, 2)** | 8370 | 8337 | 3.165 | **2.785e-4** | **0.99927** |

The last is `results/extended_ansatz.json` (re-encoded as 323 maximal shells) with its
optimum `results/extended_state.json` (1.6 h on 24 CPUs, 52 GB; measured
again with 256-bit balls).

Within this family (the 13 published seeds, Yukawa towers to total degree 21, projected
Gaussian profiles and the invariant multiples s, u, v) ⟨H_aux⟩ levels off near 2.8e-4, F_lb near
0.9993; a further step would need new covariants (seeds) rather than more profiles.

New covariants did take the next step: `results/seed-search.md` (F_lb = 0.9999978).
