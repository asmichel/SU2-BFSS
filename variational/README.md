# bfss — variational program for the SU(2) BFSS ground state

The variational construction of
[arXiv:2611.09837](https://arxiv.org/abs/2611.09837): gauge- and
Spin(9)-invariant trial wavefunctions, their matrix
elements computed exactly or in ball arithmetic of tunable precision, the
minimization of $`\langle H_{\rm aux}\rangle`$ and the measurement of states.
The bound on the ground-space fidelity is
$`F \ge [1-\kappa\langle H_{\rm aux}\rangle]_+`$ with
$`\kappa = (36+8\sqrt{14})/25`$; `bfss measure` prints
$`\langle H_{\rm aux}\rangle`$ as `Haux` and the bound on $`F`$ as `F_lb`.

A guide to the code, next to the paper, is at <https://michel.ac/SU2-BFSS/>.
[docs/data-structures.md](docs/data-structures.md) maps the types and the
state files.

## Use

Python 3.12 or later, with [uv](https://docs.astral.sh/uv/):

```sh
# numpy, scipy, python-flint, pytest
uv sync
# the state of the paper's earlier code addendum -> state file
uv run bfss import-reference tests/data/reference -o reference.json
# norm, E, Haux, F_lb, <T>, <u>, <Y>, <R^2>..<R^8>, canonical blocks
uv run bfss measure reference.json
# its basis as an ansatz file
uv run bfss ansatz reference.json -o ansatz.json
# optimize from scratch, over the dilation or at a given one
uv run bfss optimize ansatz.json -o optimized.json
uv run bfss optimize ansatz.json -o at3.json --dilation 3
# also store the observables in the state file
uv run bfss measure --store at3.json
# 128-bit balls (default 256), or exact rational matrix elements
uv run bfss --prec 128 measure optimized.json
uv run bfss --exact measure optimized.json
```

`bfss contract` precomputes the contraction table of an ansatz; tables and the
operator library are cached in `.cache/` (`--cache DIR`). `--threads`,
`--workers` and `--verbose` set the FLINT threads, the contraction workers and
progress output.

On 14 CPUs the first contraction table of the published basis (1951
functions) takes about 11 min; with it cached, `bfss measure` takes 30 s and
4 GB and `bfss optimize` 2 min and 5 GB. The reported basis (19652 functions)
took 3.1 h on 48 CPUs and 113 GiB
(`bfss --prec 128 optimize results/seeded_ansatz.json -o seeded_state.json --dilation 3`).

## Tests

`uv run pytest` takes about 20 s with a warm cache and suits every change.
`uv run pytest --slow`, about 9 min, adds brute-force Fock-space and
quadrature oracles, the long algebraic identities and end-to-end checks
against the state of the earlier addendum.

## Results

| basis | functions | $`\langle H_{\rm aux}\rangle`$ | bound on $`F`$ | state |
| --- | --- | --- | --- | --- |
| earlier addendum | 1951 | $`3.4945\times10^{-3}`$ | 0.99078 | `results/optimized_state.json` |
| extended | 8370 | $`2.7845\times10^{-4}`$ | 0.999265 | `results/extended_state.json` |
| seeded (reported) | 19652 | $`8.2116\times10^{-7}`$ | 0.9999978 | `results/seeded_state.json` |

Each state sits next to its ansatz and its 256-bit measurement in `results/`
(the earlier addendum's ansatz is `results/ansatz.json`).
[results/ansatz-search.md](results/ansatz-search.md) and
[results/seed-search.md](results/seed-search.md) record the searches that
built the extended and seeded bases,
[analysis/eft-density.md](analysis/eft-density.md) the densities behind the
paper's figures, and
[tests/data/reference/README.md](tests/data/reference/README.md) the
published state evaluated exactly.
