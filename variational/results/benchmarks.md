# Benchmarks

Measured on the development machine (14 CPUs, 84 GB) unless stated otherwise.

## This package against the prototype

The prototype is the exploratory repository this package was distilled from. Both
evaluate the same published 1951-function state with analytic matrix elements.

| | prototype | this package |
|---|---|---|
| code | 350 Python files, 152,347 lines; 33 C++ files, 6,768 lines; 852 MB on disk | 17 Python files, 3,208 lines (+1,119 lines of tests) |
| basis and seeds | 13 seed IDs and their towers hard-coded in the evaluator | seeds and shells are data (`bfss.ansatz.v1`, `bfss.state.v1`) |
| matrix elements | float64 contractions, binary128 accumulation; two interpolation stencils differ by up to 1.4e-6 in the blocks, 4.4e-4 in F_lb | exact rationals, or balls of tunable precision with rigorous radii (1e-58 at 256 bits) |
| published state, cold | 52.5 min on one core, 1.7 GB, writing a 649 MB entry file (`orthogonal_projected_evaluate.py --rebuild-entries`) | 25 s operator library + 11 min contraction table on 13 workers (2.4 CPU-hours, 71 MB) + 29 s `bfss measure` |
| published state, warm | 66 s (6.6 s of contraction after loading the entry file) | `bfss measure`: 20 s (128-bit), 29 s (256-bit) |
| optimization of the 1951-function basis | 5-6 s float64 search, after an incremental preparation of 100-300 s for each extension of the basis | `bfss optimize` from scratch: 133 s with 256-bit balls, 104 s with 128-bit (14 CPUs; 196 s before the blocked Gram-Schmidt, the eigen search and the fixed-point frame transform) |
| E, F_lb of the published coefficients | 2.115e-4, 0.99043 (primary stencil); 2.173e-4, 0.98998 (alternative) | 2.83e-4, 0.98800 (exact matrices) |

The prototype's two interpolation stencils give the same coefficients F_lb values 4.4e-4 apart,
and its coefficients were optimized against those matrices; evaluated exactly they give a lower
bound by 2.4e-3. This package's matrix elements are exact (or rigorous balls), so its optimum is
a property of the basis: the same span optimized exactly gives F_lb = 0.99022, all 1951 functions
0.99078.

## The full reduction to the 2066-dimensional fibre

The test oracle `FiberContext` in `tests/test_contractions.py` (in `contract.py` until v0.6)
represents each Klein sector by an orthogonal basis (exact modulo p) of its SU(4) invariants,
the kernel of the flavour Casimir: (1,36,468,1056,468,36,1) = 2066 states in the trivial sector
and (0,30,432,1020,432,30,0) = 1944 in each other one. The live reduction (`ModularContext`)
keeps the Klein sectors of flavour-orbit space, 3967 states in the trivial sector, about twice
as many, but with sparse operators. All seven contraction families agree exactly between the
two (`test_full_reduction_to_the_fibre`).

One prime, 13 seeds, sample points (2, 3, 5) after a warm-up point:

| cores | orbit space (3967, sparse) | fibre (2066, dense) |
|---|---|---|
| 39 (powers 0-2) | 0.4 s per point | 1.2 s per point |
| 208 (powers 0-15) | 1.3 s per point | 2.8 s per point |
| setup per prime | none | 51 s (kernels and orthogonal bases) + ~20 s (densifying operators) |

Restricted to the fibre the operators fill dense 2066 x 2066 (or 2066 x 1944) matrices, so
every application costs 2066^2 per column against a few nonzeros per row in orbit space;
the halved dimension only pays in the Gram products and the few operators that are dense in
both. The sparse orbit representation stays the default. (Time box: 10 minutes of runs.)
