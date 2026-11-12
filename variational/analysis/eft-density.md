# EFT figure data

`eft_density.py` evaluates a saved state in physical coordinates. It exports the radial and joint densities,
the conditional transverse density, and the transverse moments, with the density, slice measure, quadrature and
grids of `radial_tail.py`. Each output includes the state hash and every requested quadrature order; the
conditional density is normalized by its numerical radial marginal. `assemble_eft_density.py` joins the shards
of each task, rejects shards of another state and incomplete shard sets, and records the
numerical residuals against the radial-tail record of the same state, which supplies the finer, higher-order
radial grid. No variational optimization is needed to regenerate the figures.

| state | tail record | EFT density record | evaluated at | EFT run |
|---|---|---|---|---|
| extended (8370 functions) | `extended_tail_fit.json` | `extended_eft_density.json` | `7b1c900` (tail: `0a3b8ce`) | 2026-10-22 |
| seeded (19652 functions) | `seeded_tail_fit.json` | `seeded_eft_density.json` | `b861df2` | 2026-11-05 |

The extended-state EFT record was made on the branch `regenerate-eft-plots` (assembled at `cf0251a`) and merged into master with the seeded records. `tests/test_records.py`
checks all four records against their states: the tail fits against their profiles, and each EFT section's
top-level values, order agreement, `rho2_over_eft`, `rho4_over_rho2_sq` and `p_cond`, and validation residuals against
the assembler.

## Running

These are the seeded state's runs, on 12 CPUs each, writing outside the tracked files:

```sh
run=runs/seeded-eft; mkdir -p "$run"; s=results/seeded_state.json
py() { uv run python "analysis/$1.py" "${@:2}"; }
for i in $(seq 0 7); do py radial_tail density $s "$run/tail_$i.json" --radii 3:9:0.025 --orders 48,64 --shard "$i/8"; done
py radial_tail density $s "$run/check96.json" --radii 3:9:0.5 --orders 64,96
py radial_tail norm $s "$run/norm.json"
for i in 0 1; do py eft_density radial $s "$run/radial_$i.json" --radii 0.05:3:0.05 --orders 32,48 --shard "$i/2"; done
for i in 0 1 2; do py eft_density joint $s "$run/joint_$i.json" --radii 0.5:8:0.1 --rho 0:3:0.04 --orders 24,32 --shard "$i/3"; done
py eft_density radial $s "$run/transverse_0.json" --radii 4.8:6.4:0.2 --orders 48,64
py eft_density conditional $s "$run/conditional.json" --radius 5.6 --rho 0:3:0.02 --orders 32,48
```

and then:

```sh
uv run python analysis/radial_tail.py fit "$run/fit.json" "$run"/tail_*.json --norm "$run/norm.json" --compare 4.8,6.4
uv run python analysis/curate_tail_fit.py "$run/fit.json" "$run/norm.json" results/seeded_state.json \
    results/seeded_measurement.json COMMIT results/seeded_tail_fit.json --check "$run/check96.json"
uv run python analysis/assemble_eft_density.py --radial "$run"/radial_*.json --joint "$run"/joint_*.json \
    --transverse "$run/transverse_0.json" --conditional "$run/conditional.json" \
    --tail results/seeded_tail_fit.json --out results/seeded_eft_density.json
```

The extended state's tail ran as in the README (orders 32, 48, 64 and 96 on all radii, so `fit` records the
64/96 agreement itself), and its EFT tasks with the same grids in 4 radial, 8 joint and 3 transverse shards. The
`generation` field of each record (date, code commits, commands) is written by hand.

Resources: the extended state's `Density` peaks at 12.5 GB, its EFT shards took 5–7.5 min on 8 CPUs and its tail
shards about 20 min. The seeded state's operator matrices of 991 cores take about 23 GiB; the peak RSS was
43.7 GiB and the setup about 8 min; on 12 CPUs the tail shards took 42–46 min, the 64/96 check 48 min, `norm`
77 min, the radial and joint shards 25–27 min, the transverse task 19 min and the conditional 12 min.

## Results and convergence

| | extended | seeded |
|---|---|---|
| flattest window of ratio 1.1 | 5.33–5.86 | 6.36–7.00 |
| C_fit/C at power 10, free power | 1.020, 10.11 | 1.013, 10.06 |
| local power within 2% of 10 | 4.80–6.45 | 4.78–8.93 |
| tail orders 48/64, 64/96 (max rel.) | 4.4e-6 (48/96), 7.9e-10 | 4.4e-6, 6.6e-7 (on 3:9:0.5) |
| norm − 1, ⟨R²⟩ against the measurement | −3.6e-8, −4.5e-7 | −1.4e-7, −3.1e-6 |
| EFT radial p, orders 32/48 | 1.6e-10 | 4.6e-7 |
| EFT transverse p, orders 48/64 | 1.7e-9 | 1.3e-7 |
| joint p, orders 24/32: above 1e-10, 1e-4, 1e-2 of the peak | 3.3e-7, 6.0e-9, 1.5e-9 | 1.1e-3, 1.5e-5, 2.6e-6 |
| conditional p_cond, orders 32/48: above 1e-10, 1e-4 of the peak | 4.4e-7, 4.3e-7 | 2.2e-4, 1.0e-6 |
| conditional mass − 1 (Simpson) | −2.4e-10 | −2.2e-8 |
| bulk radial against the tail at r = 3 | 5.4e-11 | 1.4e-7 |
| transverse against the tail | 1.0e-10 | 0 (not independent) |
| joint marginal against the tail (Simpson) | 2.2e-7 | 5.8e-7 |

The seeded state converges more slowly in the shape quadrature: the joint density at orders 24/32 and the
conditional density at 32/48 differ by up to 1e-3 and 2e-4 where the density is below 1e-4 of its peak, and by
at most 1.5e-5 and 1e-6 above it. The seeded tail was computed at order 64, the same order and radii as its
transverse task, so their agreement is an identity rather than a check; the extended tail used order 96. These
checks establish the numerical consistency of the plotted data; they do not estimate the variational error of
the state.

## Provenance

The records hold the derived data, the commands and the code commits; the raw outputs they were assembled from
are not in the repository.

* **Seeded records** (run by Jaroslav Scheinpflug). The evaluations ran in a plain venv with the locked numpy, scipy and
  python-flint, so the Python and package versions are not recorded. The tail record's `quadrature_check_64_96`
  (only its maximum, 6.6e-7, is kept) and both `generation` fields were added by hand; `curate_tail_fit.py` was
  committed after the run; `norm.json`'s `r2`, `r_rms` and `rho2` and `fit.json`'s window scans (except every
  fifth of ratio 1.1; they follow from the profile) were not kept. The EFT record pins the tail record's hash
  after the corrections of its `generation` field and notes the hash it was assembled from.
* **Extended EFT record**. The conditional density was evaluated separately (`generation.local_conditional`)
  and is kept, at both orders, only in the record. `validation.combined_radial_integral` was added by hand and
  its recipe is not in the repository: per its note, Simpson and trapezoid integrals of p over the bulk grid
  joined with the tail grid to r = 9.

On 2026-11-08 the keys of the records were renamed to the paper's symbols (here `A` → `rho2_over_eft`, `Kt` →
`rho4_over_rho2_sq`; in the states `D_eigenvalue` → `Haux_eigenvalue` and the observables `D`, `F` → `Haux`,
`F_lb`), which changed the bytes and hence the hashes of the states and tail records. Each record's
`state_sha256_note` gives the hash of the state file its evaluation read, and the EFT records'
`sources.tail_sha256_note` that of the tail record they were assembled from.

Recovery is in principle a rerun. The evaluations are deterministic up to floating-point rounding, which depends
on the host, BLAS and thread count: rerunning the exporter at v0.6 on another machine reproduced the extended
record's radial, transverse and conditional values to 3e-11 and its joint values above 1e-2 of the peak to
3e-10, and on a test state the refactor after `b861df2` changes the results by at most 1e-16. To recover a
record, check out its code commit (or any later one), run the commands above, curate and assemble, and compare the
result with the committed record; the fits, band, windows and validation follow from the data by the committed
scripts, and `tests/test_records.py` checks the result's consistency. The
seeded state's full rerun took 10.7 hours on 12 CPUs (128 CPU-hours) at 58 GB per task.
