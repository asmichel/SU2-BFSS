# Beyond the published seeds: more of the 2066-dimensional fibre, and the valley tail

The extended state (`results/extended_state.json`, ⟨H_aux⟩ = 2.7845·10⁻⁴, F_lb = 0.99927) uses the 13
seeds of the published basis (the paper's earlier code addendum): its trial functions are
f_{a,b} s^r u^k v^l Y^p B over 250 cores Y^p B. With those
seeds the earlier search had levelled off (`results/ansatz-search.md`). This record shows that the
seeds were the limit: new seeds, all of them expressions over the existing vocabulary (C2, Ksp, Y(X),
Y(W(X))) and therefore covariants of the paper's family by construction, lower ⟨H_aux⟩ 35-fold. Profiles that
reach further along the commuting valley then remove two thirds of what is left, and invariant multiples of the
new seeds, with more widths for some of them, more than half of the rest; a third more functions, which faster
linear algebra and a fixed dilation fitted into the same four hours of a run, another 27%: ⟨H_aux⟩ = 8.2116·10⁻⁷, F_lb = 0.9999978.
The run names below are this record's own, unrelated to those of `results/ansatz-search.md` (F1, F2, F3 name different bases there).

## Where the Yukawa towers live in the fibre

At a canonical point a state is psi(D(x)) = sum_k phi_k(x) Y(x)^p_k B_k(x) with scalar coefficient
functions phi_k: pointwise it lies in the span of the core values. Exactly (modulo a prime) at generic
points:

* the 250 cores are independent at a point: the extended state lies in a 250-dimensional subspace of
  the fibre at each point (the subspace turns with x);
* Y(D(x)) maps the fibre to itself, with 151 distinct eigenvalues (multiplicities 1 to 98, spectrum
  symmetric). Each published seed but O has a component in every eigenspace; all towers of the 13 seeds
  together, to any power, span 1102 of the 2066 dimensions;
* that span contains every eigenspace of multiplicity at most 11, which is the whole bottom of the
  spectrum: the non-degenerate Yukawa vacuum and every excitation up to about 5.6|x| above it. The 964
  missing dimensions are the degenerate middle of the spectrum;
* operators that do not commute with Y reach them: words in Y and the colour Casimir C2 applied to the
  singlet S span 2064 dimensions, Y and the spatial Casimir Ksp 2065, and the slice Yukawa components
  with C2, applied to the 13 seeds, all 2066.

The Yukawa towers explore the fibre efficiently in one sense: they reach the whole bottom of the
spectrum, which is where the fermions of the ground state mostly are (<Y> = -34.2 is close to the
Yukawa vacuum energy at the typical radius). But every tower lies in the Krylov space of Y on its seed,
and within each degenerate eigenspace 13 seeds give at most 13 directions.

## How much of the covariant module the seeds generate

The covariant module over the invariants s, u, v has 2066 generators; counted by degree (characters,
the prototype's `harmonic_degrees.py`): 1, 1, 4, 11, 24, 44, 83, 115, 165, 196, 231, 228, 235 for
d = 0..12. A tower supplies one generator per degree, so 13 seeds give at most 13 per degree: they are
complete through degree 3 and give 12 of the 24 quartic generators. The missing ones, found exactly at
generic points, are Casimir images and Y_W = Y(W(X)) words:

| degree | generators | from the 13 towers | new seeds that complete the degree |
|---|---|---|---|
| 4 | 24 | 12 | 12: C2 and C2^2 images of the bare quartics (C2 Y^4 S, C2^2 Y^4 S, ... C2 Y^2 A, C2 Y H3) |
| 5 | 44 | 25 (with the quartics) | 19: Y_W U0, Y_W L2, Y_W A and Casimir images of Y^5 S, Y^3 U0, Y^3 L2, Y^2 O |
| 6 | 83 | 44 | 39: Y_W words of the cubic seeds and Y_W Y^3 S, and Casimir images |
| 7 | 115 | 83 | 32: Y_W words of the quartic seeds and of Y^4 S, ..., and Casimir images |

The published Z, R, C are the first three nested commutators [..[Y_W, Y].., Y] S; the Y_W words above
continue that construction.

## Is the missing part of the fibre occupied? The pointwise residual

For the variational optimum psi the residual g = (H_aux - ⟨H_aux⟩) psi is orthogonal to the trial space. At
each point the part of g(x) outside the span of the core values cannot be reached by any trial function
over those cores; the part inside can, by better coefficient functions (profiles, invariant multiples).
`analysis/slice_residual.py` applies H_aux = (1/16) sum_a Q_a R^2 Q_a on the slice directly (H from the
slice Laplacian with its Jacobian and the rotation generators off the slice; frakB and frakV) and
integrates with the slice measure; on a test state it reproduces bfss measure's E and ⟨H_aux⟩ to 1e-5.

| state | int abs(g)^2 | outside the core span | absorbed by the best candidates |
|---|---|---|---|
| extended (13 seeds) | 8.75e-2 | 63% | quartic towers p <= 8: 83% of it |
| QB (+ quartic seeds) | 3.01e-2 | 23% | 12 quintic/sextic seeds: 90% |
| QA (+ quartic seeds) | 2.29e-2 | 44% | 16 quintic/sextic/septic seeds: 94% |

The residual of the extended state sat mostly outside its fibre directions, where the probability is
(|x| between 2.5 and 4). The greedy ranking of candidates by the residual their towers absorb (`rank`)
chose the sextic and septic seeds below. (A cheaper proxy, the local ground state of Y plus the
angular kinetic terms at fixed x, lies in the core span to 1e-7..1e-4: the missing directions come from
the derivative couplings of H, not from the local fermion energy, and that proxy is misleading.)

## Variational tests

Every number is an optimization from scratch at 128 bits (`bfss optimize`; ⟨H_aux⟩ from the run's own ball
measurement, radii 1e-9..1e-16: long towers of degree-7 seeds strain 128-bit balls, U1's radius is 5e-4 of ⟨H_aux⟩;
the seeded state is measured again at 256 bits). P = the published widths (2,1),(3,1),(4,1); C = (2,2),(3,2),(4,2),(5,½);
"all" = the 12 widths of the extended ansatz; B0 = the extended ansatz without its bare degree-20/21 tower
ends and v multiples at the core widths (7308 functions).

| run | basis | N (kept) | λ | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|---|
| extended | 13 seeds (results/extended_state.json) | 8370 (8337) | 3.1645 | 2.7845e-4 | 0.99927 |
| QC | extended + 12 quartic seeds, p ≤ 4, all widths | 9090 (9057) | 2.9652 | 8.9696e-5 | 0.99976 |
| QB | extended + quartic p ≤ 15 at P | 8946 (8909) | 3.1882 | 7.3371e-5 | 0.99981 |
| QA | extended + quartic p ≤ 8, all widths | 9666 (9633) | 3.1639 | 5.7826e-5 | 0.99985 |
| QAp | QA with the B0 cuts (pruning test) | 8604 (8591) | 3.1129 | 6.2142e-5 | 0.99984 |
| R2x | QB + quartic p ≤ 15 at C | 9714 (9672) | 3.0309 | 5.2924e-5 | 0.99986 |
| R2y | QB + s, u, su × quartic p ≤ 7 at P | 9810 (9773) | 3.2401 | 5.2341e-5 | 0.99986 |
| QL | extended + quartic p ≤ 15, all widths | 10674 (10629) | 3.1955 | 5.0431e-5 | 0.99987 |
| S3b | QA + 19 quintic seeds p ≤ 14 at P | 10521 (10478) | 3.1228 | 1.6790e-5 | 0.999956 |
| S4G | QB + 5 quintic + 12 sextic seeds p ≤ 8 at P | 9405 (9368) | 3.1086 | 1.7353e-5 | 0.999954 |
| S4B | QA + the same 17 at P | 10125 (10092) | 3.1277 | 1.1706e-5 | 0.999969 |
| S4C | QAp + the same 17, p ≤ 8, all widths | 10440 (10427) | 2.9183 | 9.7797e-6 | 0.999974 |
| J1 | B0 + quartic p ≤ 15 at P, C + 19 quintic p ≤ 14 at P | 9507 (9448) | 3.0015 | 1.8736e-5 | 0.999951 |
| J2 | J1 + s, u, su × quartic p ≤ 7 at P | 10371 (10307) | 3.0884 | 1.6866e-5 | 0.999956 |
| U1 | B0 + quartic p ≤ 15 at P, C + 19 quintic, 14 sextic, 8 septic seeds, p ≤ 14, 13, 12 at P | 10407 (10345) | 3.0138 | 9.984e-6 | 0.999974 |
| W1 | S4C + 8 septic seeds p ≤ 8 at P | 10656 (10643) | 2.9334 | 8.7687e-6 | 0.999977 |
| W2 | S4C + the other 14 quintic seeds p ≤ 8 at P | 10818 (10805) | 2.9268 | 8.2734e-6 | 0.999978 |
| X1 | W2 + the 8 septic seeds p ≤ 8 at P | 11034 (11021) | 2.9321 | 7.9281e-6 | 0.999979 |
| X2 | X1 − u, su multiples at the core widths with b = 2 (1344) + septic and 14 quintic seeds p ≤ 6 at the other 9 widths | 11076 (11075) | 3.0017 | 8.3844e-6 | 0.999978 |

QA, QB, QC, U1, W1, W2, X1 and X2 ran on 48 CPUs (U1 to X2: 2:50-3:23 each); the others in pairs on 24 CPUs
each. U1 and a variant with quartic multiples (U2) first ran out of 75 GB each in the frame
transforms at about 10500 functions; U1, W1 and W2 were rerun alone (the code now transforms one block at a time
and builds the fixed-point matrices row by row). The pruning test QAp shows
that the B0 cuts cost 7.5% of ⟨H_aux⟩ for 1062 functions, little against what new-seed functions buy.


## Larger bases

At 11000 functions an optimization spent 2.3 of its 3.4 hours in three fixed-point frame transforms. FLINT's
multimodular integer products slow down per operation beyond a few thousand rows (on 48 threads 0.17 ns per n³ at
2000 rows, 0.34 at 4000, about five times the 2000-row rate at 11000). The transform now multiplies blocks of 2048 rows, skips
the zero blocks of the triangular L⁻¹ and computes only the lower blocks of the symmetric result, with bit-identical
output: a transform at 11930 functions took 19 min instead of about 57. The dilation search hid a cost of the same
size: once ⟨H_aux⟩ is ~10⁻⁶ of the largest eigenvalue of the residual form, the lowest Ritz value of the subspace
iteration reaches the float64 rounding floor in three or four steps and then wanders above the stopping tolerance,
so the dilations ran to the iteration cap (the dilation search took 30-52 min for bases with tail profiles); the
iteration now also stops when the Ritz value no longer decreases (7-9 min). About 15000 functions fit in the four
hours of a run.

## The valley tail

With W2's seeds and the eight septic Y_W words (X1) the new seeds had done most of their work. The residual split
of X1 leaves 1.3% of ∫|g|² outside the span of its cores (S4C: 6.8%) and 11% where the probability is (S4C: 37%);
the two of the 14 new sextic seeds it lacks would absorb a fifth of that. More structure on the fibre side gave
little: s, u, su multiples of the new towers at the published widths (Y2) removed 16% of ⟨H_aux⟩ with 2754 functions,
longer new towers and more widths (Y3) 13% with 2214. What remained lies inside the span of the cores, mostly far
along the valley (84% of ∫|g|² at |x| > 4.5; |g|² weights the tail by powers of r and overstates it).

On the valley a profile is f_{a,b} ≈ exp(−b r²/2 − a t/2) (t transverse): a width serves radii r ~ a, where its
transverse width matches the W bosons, as long as b a²/2 is of order one there. The broadest profile of the
extended ansatz, (8, ¼), ends near r ≈ 6.4 (the extended state follows r⁻¹⁰ to there), but the exact density
p ≈ C r⁻¹⁰ (C = 1890√2) puts P(r > R) = (C/9) R⁻⁹ = 7.4·10⁻⁶ beyond R = 7 and 2.2·10⁻⁶ beyond 8, not small
against κD = 2.1·10⁻⁵ of X1. Profiles further out are members
of the paper's family ("smaller values of b broaden the longitudinal profile and extend the accessible valley
range"):

| run | basis | N (kept) | λ | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|---|
| Y2 | X1 + s, u, su × the 51 new towers p ≤ 5 at P | 13788 (13775) | 2.9216 | 6.6267e-6 | 0.999983 |
| Y3 | X1 + new towers longer at P (quartic p ≤ 15 also at C) + 14 quintic and 8 septic seeds at C + the 2 other sextic seeds at all widths | 13248 (13165) | 2.9187 | 6.9120e-6 | 0.999982 |
| Y1 | X1 + the 224 bare cores of the published seeds at (10, ¼), (12, ¼), (8, ⅛), (10, ⅛) | 11930 (11917) | 2.9993 | 2.4423e-6 | 0.9999936 |
| Y4 | Y1 + the same at (12, ⅛), (16, ⅛), (10, 1/16), (14, 1/16) + s × them at (8, ⅛), (10, ⅛) | 13274 (13260) | 3.0012 | 2.0359e-6 | 0.9999946 |
| Z1 | Y1 + the same at (6, ⅛), (8, 1/16), (10, 1/16), (12, 1/16), (12, 1/32), (16, 1/32) | 13274 (13261) | 2.9985 | 2.0330e-6 | 0.9999946 |
| Z2 | Z1 − the new seeds at (3, ½), (6, ½), (5, ¼), (8, ¼) (1044) + s × the bare cores at Y1's four widths | 13126 (13110) | 2.9064 | 2.4632e-6 | 0.9999935 |
| F1 | Y1 + the 64 tail cores below, bare and times s, at 18 tail widths (a = 6-24, b = ⅛-1/64) | 14234 (14221) | 2.9951 | 2.0278e-6 | 0.9999947 |

Four tail widths with 896 functions removed 69% of ⟨H_aux⟩; the radial moments grew with the tail (⟨R⁸⟩ from 6109 to
6186). Four or six more widths, down to b = 1/32 and out to a = 16, or s multiples at the tail widths, removed 17%
more however they were placed, and paying for them with the new seeds at four valley widths cost more than it gave.

The tail needs few cores. At valley points (r = 4-10, small transverse singular values) Y1's state lies, on average
to 10⁻⁶ of its norm squared (10⁻⁵ at the worst point), in the span of 64 cores of the published seeds (`slice_residual.py cores` in v0.6, since removed; a greedy
simultaneous selection): high powers of any tower converge to the Yukawa vacuum direction, two powers of one seed
already leave about 1%, and a few dozen more cores give the corrections. Those 64 cores at 18 more tail widths (F1) do
no better than all 224 at four or six (Y4, Z1): beyond Y1 the tail is saturated. The residual split of Y4 agrees:
∫|g|² fell to 1.6·10⁻³ (X1: 7.2·10⁻³), its part beyond |x| = 9 by a factor of 150, and 65% of it is now at
|x| < 4.5.

## With the tail in place

The residual left is in the core region again, mostly inside the span of the cores, and the fibre-side additions
that gave 13-16% on X1 now give far more:

| run | basis | N (kept) | λ | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|---|
| F3 | Y1 + Y3's additions | 14144 (14060) | 2.9861 | 1.5344e-6 | 0.9999960 |
| F2 | Y1 + s, u, su × the 51 new towers p ≤ 4 at P | 14225 (14212) | 3.0333 | 1.2594e-6 | 0.9999967 |
| H1 | F2 + quartic towers p 9..15 at P and C | 14813 (14785) | 3.0249 | 1.1431e-6 | 0.9999970 |
| H2 | F2 + quintic towers p 9..14, sextic p 9..13, septic p 9..12 at P | 14843 (14818) | 3.0739 | 1.1578e-6 | 0.9999969 |
| H3 | F2 + the 14 quintic seeds p ≤ 8 at C + the last two sextic seeds p ≤ 8 at all 12 widths | 14945 (14932) | 3.0179 | 1.1255e-6 | 0.9999970 |

At 14800-14900 functions the remaining fibre-side additions on F2 gave 8-11% each. Two reallocations at about
15100 functions test whether choosing which functions to keep pays more than adding:

| run | basis | N (kept) | λ | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|---|
| P1 | H3 − the bare cores of the published seeds at (3, ½), (6, ½) (448) + quartic towers p 9..15 at P and C (588) | 15085 (15058) | 2.9994 | 1.1035e-6 | 0.9999971 |
| P2 | H3 − v, sv, uv, suv times the published cores at P (472) + quintic p 9..14, sextic p 9..13, septic p 9..12 at P (618) | 15091 (15065) | 3.0360 | 1.1704e-6 | 0.9999969 |

At a fixed size a reallocation moves ⟨H_aux⟩ by a few percent either way: the v multiples are worth more than longer
towers, the bare published cores at (3, ½), (6, ½) a little less than the quartic towers.

## Beyond 15000 functions

Three designs of about 16000 functions (F2 + F3; F2 + s, u, su × the new towers p ≤ 2 at C; Y1 + multiples p ≤ 6
and longer new towers at P) first never left the Gram-Schmidt frame: at 16000-16400 functions it ran over 1.5 h
where it had taken 34 min at 14200 and 39 min at 14800, too long for the 4-hour limit (they were stopped). A
profile of the frame found the cause in FLINT's ball products of factors whose entries span many binades, as
those of the inverse factors and of L21 do: 0.85 ns per n³ at 4100 rows against 0.24-0.34 for random matrices, and
the top-level product of 8200 rows ran 28 min without finishing. The recursion now takes its large products (all
sides from 6144 rows) in the equilibrated fixed point of the frame transforms, by blocks of 2048 rows; the frame of
16000 functions then took 58-61 min. The transform computes only the blocks of L⁻¹B that the lower half of
L⁻¹BL⁻ᵀ involves (a fifth fewer products, bit-identical), and the optimizer releases the Gram matrix once the
frame has it.

| run | basis | N (kept) | λ | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|---|
| G1 | F2 + F3 (the multiples, longer towers and widths together) | 16439 (16350) | 3.0265 | 9.4586e-7 | 0.99999751 |

With the dilation search (three transforms) about 16500 functions fit in a run. `bfss optimize --dilation λ`
transforms U_λ⁻¹ H_aux U_λ alone, one block instead of three: R2, G1 + s, u, su times the new towers p ≤ 2 at C (1836) +
s, u, su times them p 5..6 at P (918) + v times them p ≤ 2 at P (459), 19652 functions, then fits (frame 90 min,
the three parts of H_aux 22 min, one transform 71 min: the eigenvalue after 3.1 h; 129-136 GiB with the Gram matrix
cached, 113 GiB without it (R2c)). It ran at three dilations around G1's optimum (⟨H_aux⟩ of G1 and R2c is the optimizer's
eigenvalue, their runs having ended before or during their measurement; R2a and R2b were measured afterwards with
`bfss --prec 128 measure --store`):

| run | λ (fixed) | N (kept) | ⟨H_aux⟩ | F_lb |
|---|---|---|---|---|
| **R2a** | **3** | **19652 (19563)** | **8.2116e-7** | **0.99999783** |
| R2c | 3.0265 | 19652 (19563) | 8.2268e-7 | 0.99999783 |
| R2b | 3.05 | 19652 (19563) | 8.2472e-7 | 0.99999782 |

For the optimum x at a fixed λ the Hellmann-Feynman theorem gives the slope of the smallest eigenvalue w(λ) from
the measured canonical blocks of the state: w′ = −3λ⁻⁴⟨Haux_u⟩ − (3/2)λ⁻⁵ᐟ²⟨Haux_Y⟩, +0.140 w per unit of ln λ at 3.00 and
+0.361 w at 3.05. A parabola in ln λ through the values and slopes of R2a and R2b has its minimum at λ = 2.969, w* =
8.2064·10⁻⁷, with λ²w″/w = 13.5, and predicted R2c to 0.001%: the optimum over the dilation is shallow (an error of
1% in λ costs 0.07% of ⟨H_aux⟩), and R2a is 0.06% above it. A fourth run at 2.97 could not gain more than that.

## The final state

R2a is `results/seeded_ansatz.json` and `results/seeded_state.json`, measured at 256 bits in
`results/seeded_measurement.json` (radii below 1e-44): ⟨H_aux⟩ = 8.2116·10⁻⁷, F_lb = 0.9999978, E = 4.17·10⁻⁸, <R^2> = 8.07419,
<R^4> = 68.7529. Its 19652 functions run over 991 cores of all 66 seeds of its seed set, the 13 published ones and
53 new ones, at 16 widths. The new seeds:

* quartic (12): C2 and C2^2 times Y^4 S, Y^2 U0, Y^2 L2, Y Z, Y Phi, and C2 Y^2 A, C2 Y H3;
* quintic (19): Y_W U0, Y_W L2, Y_W A; C2, C2^2, C2^3, Ksp and Ksp C2 times each of Y^5 S, Y^3 U0, Y^3 L2;
  and C2 Y^2 O (the exact degree-5 screen);
* sextic (14): C2^3 Y (Y_W L2), Y_W Y^3 S, C2 Y (Y_W U0), Y_W H3, Ksp Y (Y_W U0), Y_W Z, Y_W Y A, Ksp Y (Y_W A),
  C2^3 Y^4 U0, C2 Y^4 U0, Y_W H2, Y_W H1, Y_W H0, Ksp Y (Y_W L2);
* septic (8): Y_W Y^4 S, Y_W (C2 Y Z), Ksp Y^2 (Y_W U0), Y_W Y^2 A, Ksp Y^2 (Y_W L2), C2^3 Y^2 (Y_W U0), Y_W Y Z,
  Y_W Y^2 L2;

all defined as expressions in the seed set of the ansatz file (Y_W = Y(W(X))). The shells:

* the extended ansatz's shells of the published seeds without their bare degree-20/21 tower ends and the v multiples
  at the core widths (B0), and their 224 bare cores at the tail widths (10, ¼), (12, ¼), (8, ⅛), (10, ⅛);
* the quartic seeds, the quintic seeds Y_W U0, Y_W L2, Y_W A, Ksp Y^3 U0, Ksp Y^3 L2 and the first twelve sextic
  seeds with towers p ≤ 8 at the 12 widths of the extended ansatz; the other 14 quintic seeds and the septic seeds
  at the published and core widths (P and C), and the last two sextic seeds at all 12 widths, all with towers
  p ≤ 8;
* longer towers at P: quartic p 9..15 (also at C), quintic p 9..14, the first twelve sextic seeds p 9..13, septic
  p 9..12;
* s, u and su times the towers of the 51 new seeds other than the last two sextic ones, p ≤ 6 at P and p ≤ 2 at
  C, and v times them, p ≤ 2 at P.

Reproduce: `bfss --prec 128 optimize results/seeded_ansatz.json -o state.json --dilation 3` computes the
contraction table of its 991 cores first (the table of all 1027 cores of the seed set took 65 min on 40 workers,
`bfss contract`); the optimization took 3.1 h on 48 CPUs and 113 GiB (129 GiB before the optimizer
released the Gram matrix after the frame; with the dilation search this size does not fit in four hours), and the
256-bit measurement 1.5 h on 48 CPUs and 77 GiB. The observables stored in the state come from a 192-bit measurement
(`bfss --prec 192 measure --store`, 1.5 h on 48 CPUs): the quadratic forms of this basis cancel so strongly that at 128 bits
the midpoints of E and ⟨H_aux⟩ were good to only 3e-11 and 5e-12 of the 256-bit values (inside their radii, 1e-11 and
3e-10).

## What it shows

* The seeds, not the profiles, limited the extended state. Its residual lay 63% outside its fibre
  directions; the missing quartic generators alone cut ⟨H_aux⟩ by 68-79% (QC, QB, QA), and the quintic, sextic and septic Y_W words
  and Casimir images by a further factor of seven.
* The Yukawa towers do explore the fibre well where the fermions of the ground state are, at the bottom of
  the Y spectrum, but a tower stays in the Krylov space of Y on its seed. The ground state also needs the
  directions that the kinetic term generates: the gradient terms of H insert Y_W into Yukawa words (the
  published Z, R, C are the first of these), and its Laplacian produces Casimir-filtered words. These are
  the seeds that pay off, in the order the residual ranking predicts.
* Once the new seeds had done their work (1.3% of the residual outside the span of the cores), the profiles
  were the limit again, this time far along the valley: four profiles reaching further out (narrower across it,
  broader along it, or both) removed 69% of ⟨H_aux⟩ with 896 functions, and the tail then saturates. The tail needs few cores: high Yukawa
  powers converge to the vacuum direction.
* With the tail in place, coefficient functions as rich as the published seeds' pay off for the new seeds too:
  invariant multiples (−48% on Y1) and long towers and more widths (−37%), and on F2 another 8-11% each.
* The four-hour limit of a run sets the size. Blocked fixed-point products in the frame transforms and, from 6144
  rows, in the Gram-Schmidt recursion, the rounding-floor stop of the eigen iteration and a fixed dilation (the
  optimum over it is shallow: 1% in λ costs 0.07% of ⟨H_aux⟩) raised it from about 11000 functions to 19652 (3.1 h on
  48 CPUs, 113 GiB). Adding functions still pays, more slowly: from 15085 to 16439 functions ⟨H_aux⟩ fell 14%,
  from 16439 to 19652 another 13%. At a fixed size, choosing which functions to keep moved ⟨H_aux⟩ by a few percent.
