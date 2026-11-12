# Results of the complete basis (2026-10-19)

Hardware: two-socket Intel Xeon (Sapphire Rapids) servers with 112 cores and 1-2 TB of memory,
except the level-8 compile (a machine with 188 GB); cores per run below. The run data in
`results/L*` (manifests, sector counts, certified tips, regions) comes from these runs.

## Hierarchies

| level | sectors | multiplets | rows | operators in rows | blocks (largest) | Ward forms | forms | rank | Ward rank | coordinates | rows removed (facial) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6  | 39  | 40  | 40  | 7,288     | 25 (5)   | 12  | 83    | 31  | 15  | 15  | 2  |
| 8  | 129 | 168 | 168 | 95,173    | 60 (13)  | 86  | 581   | 109 | 59  | 49  | 10 |
| 10 | 362 | 696 | 687 | 1,031,441 | 114 (31) | 959 | 5,517 | 434 | 268 | 165 | 57 |

Every multiplicity equals the Weyl character's. After facial reduction the largest blocks have
5, 11 and 25 rows. Ward forms at level 10: J 407, gauss 317, Q 165, H 70.

## Cost

| level | compile | peak per worker | scans |
|---|---|---|---|
| 6  | 10 s, 24 workers | 0.5 GB | 330 solves, 1024 bits, 1 core each, median 27 s; 2.7 core-h |
| 8  | 21 min, 32 workers | (not logged) | 709 solves, 1024 bits, 1 core each, median 32 min; 300 core-h |
| 10 | first prime (K = 2048, rank 434) 14.0 h on 64 workers; six primes at K = 498 as parallel jobs, ~5 h each on 48 workers; assembly 12 min (10 min hierarchy, 1 min reduction); 23 h wall in all | 46 GB | 333 solves, 768 bits, 8 MPI processes each, max 350 iterations, median 2.2 h; 711 solve-h = 5,700 core-h, three servers (one per plane), 14 solves side by side |

Single solves at level 10: minimum of <R^2> 3.0 h with 32 processes (1000 iterations); the
certification solves 600 iterations, ~1.7 h each with 32 processes. Level 12 not attempted.

## Minimum of <R^2>

| level | SDPB | certified bound (beta - abs(r)_1 Z, Z = 1e40) | residual abs(r)_1 | status | t* |
|---|---|---|---|---|---|
| 6  | 6.851889192571592239 | 6.8518891925715915 | 7.9e-59 | optimal, gap 2e-29 | - |
| 8  | 7.498628730900346575 | 7.498628730900346 | 5.2e-57 | iteration cap, gap 1.3e-16 | 0.90 |
| 10 | 7.798212232945515724 | 7.798212232943035 | 5.2e-56 | iteration cap, gap 4.1e-12 | 0.49 |

The returned points are feasible (smallest block eigenvalue from the exact data 4.4e-31,
5.7e-18, 1.9e-13).

## The variational state against level 10

Euclidean-projected variational wavefunction: <R^2> = 8.07419(1), <u> = 17.1124560(1),
<R^4> = 68.7529(1). Certified level-10 support lines closest to it (window-normalized):

| plane | certified | at the state | margin | in sigma |
|---|---|---|---|---|
| (R2, u)  | 0.08300469 R2 - 0.05241154 u >= -0.2481979 | -0.22669 | 0.0215 | 8.9 |
| (R2, R4) | -0.12796357 R2 + 0.00751553 R4 >= -0.5209232 | -0.51649 | 0.0044 | 1.5 |
| (R4, u)  | 0.00385444 R4 - 0.05502362 u >= -0.6828783 | -0.67659 | 0.0063 | 2.5 |

Each line is certified with residual 5.2e-56; certificates use directed rounding.

## Regions

| level | tolerance | worst vertex (R2_u, R2_R4, R4_u) | lower-level bounds inherited |
|---|---|---|---|
| 6  | 1e-3 | 9.4e-4, 9.8e-4, 8.2e-4 | - |
| 8  | 1e-3 | 8.0e-4, 9.8e-4, 9.8e-4 | 2, 0, 2 |
| 10 | 3e-3 | 1.8e-3, 1.5e-3, 2.5e-3 | 58, 99, 116 |

Worst violation of any bound by a feasible point inside the window: 2.2e-6 (L10, points with
coordinates near 1e50 on non-attained boundaries; tolerated up to 1e-5). Level-to-level
tightening, in units of the window height: (R2, u) lower u edge +15-22% from L8 to L10, upper
-6-14%; (R2, R4) lower R4 edge up to +13% at large R2, +0.5% near the state; (R4, u) both edges
6-9%. Figure: `figures/regions_L6_L8_L10.png`.

## Checks

| check | status |
|---|---|
| model identities (136 {Q,Q}, Fierz, commutators) | passed |
| multiplicities vs Weyl character | L6, L8, L10: all sectors |
| check_pairing (conversions, product, adjoint, pairings, invariance, null multiplets L8) | passed |
| check_problem (explicit pairings per block) | L8 60/60; L10 71 of the 73 blocks with dim <= 3000, 0 failures; the two largest (dim 2457, 2079) did not finish in the 24 h limit |
| reconstruction | verified on a held-out prime with the first 7 primes at every level |
| strict interior after facial reduction | t* = 0.90 (L8), 0.49 (L10) |
| certificates | tips L6-L10, three L10 support lines |
