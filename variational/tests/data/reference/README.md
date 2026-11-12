# Published companion data

`coefficients.json`, `fermion-data.json` and `moments.json`, copied verbatim
from the `filecontents*` environments of the paper's code addendum
(`drafts/code-addendum.tex` in BFSS-compression, commit eb0f7a7, 2026-10-18),
so that the tests need nothing outside this repository.
`bfss import-reference tests/data/reference -o reference.json` reads them as
well.

sha256:

```text
446eb7ec4ad9cafec2672d5b673713272f01b73de52bb8ddad5d481e042c888b  coefficients.json
fb28c19deb795d3b48cc7b3f5b03cd5684aee90744374003dfc5723eb52c344c  fermion-data.json
0206948e21ced0e92ede7017827f84ffec7b3adeed0b8d4a9af703e577afbc97  moments.json
```

## Evaluated exactly

Normalized blocks at $`\lambda = 1`$ (`bfss measure`; exact and ball matrix
elements agree in all shown digits), with the names that `moments.json` gives
them:

| block | in `moments.json` | exact (this code) | published primary | published alternative |
| --- | --- | --- | --- | --- |
| `T` | `T` | 11.79986579 | 11.79986149 | 11.79987272 |
| `u` | `W` | 35.95493085 | 35.95489555 | 35.95488806 |
| `Y` | `Y` | −41.19496780 | −41.19502601 | −41.19503437 |
| `Haux_T` | `DK` | 149.2898866 | 149.2897925 | 149.2899554 |
| `Haux_u` | `DV` | 454.9114953 | 454.9106716 | 454.9106403 |
| `Haux_Y` | `DY` | −521.1971413 | −521.1981128 | −521.1980861 |
| `R2` | `R2` | 11.69259456 | 11.69259159 | 11.69259176 |

The exact values differ from the primary evaluation by
$`2.5\times10^{-7}`$–$`1.9\times10^{-6}`$, as much as the primary and
alternative evaluations differ from each other. Because $`E`$ and
$`\langle H_{\rm aux}\rangle`$ cancel by five orders of magnitude, the
published coefficients evaluated exactly give $`E = 2.83\times10^{-4}`$,
$`\langle H_{\rm aux}\rangle = 4.55\times10^{-3}`$ and $`F \ge 0.9880`$
instead of the reported 0.9904.

Optimized from scratch (`bfss optimize`):

| setting | $`\lambda^*`$ | $`\langle H_{\rm aux}\rangle`$ | bound on $`F`$ |
| --- | --- | --- | --- |
| published matrices, 1913 functions (reported) | 1.44978 | $`3.6305\times10^{-3}`$ | 0.99043 |
| exact matrices, the same 1913 functions | 1.43185 | $`3.7093\times10^{-3}`$ | 0.99022 |
| exact matrices, all 1951 functions | 1.40795 | $`3.4945\times10^{-3}`$ | 0.99078 |
| published coefficients, evaluated exactly | (1.44978) | $`4.5514\times10^{-3}`$ | 0.98800 |

On the published span the exact optimum lies $`2\times10^{-4}`$ below the
reported bound, within the $`4.4\times10^{-4}`$ sensitivity the published data
report for their own evaluation. The exact Gram matrix has full rank: the 38
functions the published state omits have Gram–Schmidt residuals between
$`10^{-9.3}`$ and $`10^{-16.4}`$, so the omissions reflect the evaluation
accuracy, not linear dependencies.
