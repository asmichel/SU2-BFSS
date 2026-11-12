# Data structures

Each entry is a base class with its subclasses (←), and the module in
`src/bfss/` that defines it.

* `HilbertVector` (element of the singlet space) ← `TrialFunction` (a trial
  wavefunction $`\psi_j = f_j P_j`$: a `Profile` times a polynomial singlet)
  and `Expansion` (finite combination over a `Basis`); `hilbert`.
  `Expansion.dilated` applies the unitary dilation $`U_\lambda`$, which is how
  states are measured and how optimized coefficients are placed at
  $`\lambda^*`$.
* `Profile` ← `ProjectedGaussian`; `hilbert`.
* `Scalars` (number system of the matrix elements) ← `Rationals` (exact) and
  `Reals(prec)` (balls of `prec` bits); `scalars`. The same assembly, masters
  and observables run in either.
* `LinearCombination` (element of a free module over a coefficient ring) ←
  `Polynomial` (three variables; ← `Poly` in `covariants`, with
  $`\mathbb Q(\zeta_8)`$ coefficients on the slice) and `MatrixPolynomial`
  ($`a\mathbf 1 + bA + cA^2`$ over the invariants of $`A`$: Wishart moments
  with $`A = K^{-1}`$, integrands with $`A = G`$), both in `algebra`, and
  `FermionOperator` (normal-ordered words) in `fock`. The subclasses differ
  only in their coefficient ring and product.
* `PolynomialSinglet` ← `Seed` ← {`TensorSeed`, `ExpressionSeed`}, and
  `PolynomialSinglet` ← `OperatorImage` (image under an equivariant operator
  field) ← {`YukawaPower`, `InvariantMultiple`}; `covariants`. `SeedSet` holds
  the seeds of a basis.
* `Operator` (the expression language of `ExpressionSeed`) ← `ScalarOp`,
  `BaseOp`, `Product`, `Sum`; `covariants`.
* `QuadraticForm` ← `Block`, `Observable`, and `Piece` (integrand of a
  `Block`) ← `Insertion`, `Gradient`, `OneSidedB`; `blocks`, which also holds
  the `Assembler` of matrix elements.
* `SliceArithmetic` (operators on the gauge slice over a field) ←
  `ComplexContext` in `fiber`, and `ModularContext` ($`K_4`$ sectors of orbit
  space, modulo a prime) in `contract`.
* `Ansatz` (ordered `Shell`s) builds a `Basis`; `ansatz`. `Engine` in `state`
  caches operators and contraction tables.

## Files

A `bfss.state.v1` JSON file (`state`) holds the seed definitions, the ordered
trial wavefunctions (widths `a`, `b`, invariant exponents $`(k, l, m)`$, Yukawa
power $`p`$, seed name), their coefficients as decimal strings, and metadata:
`dilation` ($`\lambda`$), `canonical_width`, `Haux_eigenvalue` (the
optimizer's smallest eigenvalue $`\varepsilon`$, the minimal
$`\langle H_{\rm aux}\rangle`$), `fidelity_lower_bound`
($`[1-\kappa\varepsilon]_+`$), `matrix_elements` (the number system) and,
after `bfss optimize` or `bfss measure --store`, `observables` (the midpoints
printed by `bfss measure`). Every building block serializes itself, so loading
rebuilds the wavefunction with the generic machinery.

A `bfss.ansatz.v1` file (`ansatz`) describes a basis as shells: widths ×
invariant prefactors × seeds × ranges of Yukawa powers.
