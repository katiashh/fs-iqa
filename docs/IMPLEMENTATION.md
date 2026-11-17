# Implementation and reproducibility notes

This release packages the provided paper, appendix, DBCNN demonstration, and
1,000-row KonIQ results export. The original Python file is retained byte for
byte in `reference/`. The `fs_iqa/` package is an edited, testable inference
implementation. It is not a recovered full experimental codebase.

## What is implemented

| Component | Included implementation |
|---|---|
| DBCNN feature extractor | VGG + SCNN bilinear features, matching the supplied class |
| FTN | Exact supplied layer shapes; sigmoid output in `[0, 1]` |
| Scorer | 512 → 256 → 128 → 64 → 1, with the supplied BatchNorm/LeakyReLU blocks |
| Median prediction | Batched Gaussian feature sampling; optional clipping |
| Feature interval | Exact binomial order-statistic selection with joint confidence |
| IVE | Full Jacobian spectral norm and the supplied directional curvature proxy |
| Training / attacks / other backbones | Described by the paper; not supplied as executable implementations |

## Changes relative to the supplied demo

| Area | Original demo | Cleaned implementation |
|---|---|---|
| Imports and execution | Uses `cv2` without importing it; loads models at import | Explicit dependencies; execution behind a CLI main guard |
| Paths | Fixed paths under `data/` | Command-line paths and checks before model construction |
| Model loading | Duplicate backbone/FTN instances; separate VGG and SCNN loading | Shared modules; full DBCNN state; meta-device initialization |
| Backbone mode | Not explicitly switched to eval | All modules use eval; parameters frozen, input gradients retained for IVE |
| Score sorting | `.sort()` on `(N, 1)` sorts the singleton dimension | Flatten scores and sort across Monte Carlo samples |
| Confidence indices | Unscaled normal-CDF expression; potentially invalid indices | Exact binomial tails; missing endpoints yield abstention |
| Sample count | Discards a non-full final batch | Uses exactly the requested sample count |
| Feature radius | Interval uses 0.005, IVE uses σf = 0.1 | Both use δf = κσf |
| Image dimensions | OpenCV `(384, 512)` = W384 × H512 | Explicit W512 × H384 defaults; configurable |
| Randomness | No seed set in the file | Recorded seed and stable per-image sampling |
| IVE edge cases | Floors a small Jacobian norm | Abstains below the configured threshold |
| Quadratic root | Subtracts nearly equal floating-point values | Algebraically equivalent stable positive-root formula |
| Output | Rewrites the growing CSV on each image | Appends each row; records IDs, parameters, completion status |

These changes affect numerical outputs. The bundled CSV and paper tables are
archival results; they have **not** been recomputed with the cleaned code.

## Feature-space intervals

For noise σf and feature radius δf, define p− = Φ(−δf/σf) and
p+ = Φ(δf/σf). For N sorted samples and failure budget
β = 1 − confidence, choose one-based ranks kL, kU satisfying:

```math
\Pr[\mathrm{Bin}(N,p_-) < k_L] \le \beta/2,
\qquad
\Pr[\mathrm{Bin}(N,p_+) \ge k_U] \le \beta/2.
```

The largest valid kL and smallest valid kU give the tightest available
endpoints for this equal-tail failure allocation. The union bound gives
simultaneous confidence at least `confidence` for the population quantile
bounds. This confidence is per input, not a dataset-wide guarantee.
If a finite endpoint cannot be selected, it is left blank and
`interval_status=abstained`. Scores remain available.

The confidence statement concerns the ideal population median of the smoothed
predictor under feature perturbations. It does not mean that every independent
finite-sample prediction must lie in the interval. The reported score is a
sample median; for even N, the cleaned implementation averages the middle two
samples. The original uses `torch.median`, which chooses the lower middle sample.

The supplied demo clips noised features to `[0, 1]`; this is retained by default.
For the feature-space argument, clipping is a deterministic part of the base
scoring function: h(z) = Scorer(clip(z)). Gaussian noise is added before this
function. `--no-feature-clipping` uses the unclipped expression in the paper;
it changes the predictor and should be reported with results.

## Input-space interpretation

The main paper explicitly distinguishes conditional input-space guarantees
from empirical IVE estimates (Theorem 1, Proposition 1, and Section 6.4).
A deterministic input certificate would require valid sensitivity bounds
throughout the entire perturbation ball and the theorem's regularity conditions.

The supplied fast routine has an additional approximation: it differentiates
`B(z).sum()` and estimates a directional finite difference of that gradient.
This measures curvature of the **sum of feature outputs**, not the operator
norm of the vector-valued Hessian. Feature contributions may cancel, and one
sampled direction need not maximize curvature. Taking the maximum over sampled
values does not turn this statistic into a worst-case Hessian bound.

Accordingly, the cleaned output calls it `curvature_proxy` and the resulting
radius `epsilon_x_estimate`, with `ive_status=empirical_proxy`. The original
`vol_bound = log(1/alpha)/N` is not exported as a guarantee for this proxy.
The archived CSV retains its original column names and values unchanged.
Feature-space intervals and input-space estimates must not be conflated.

## Validation scope

CPU tests exercise exact tail probabilities, empirical coverage for a known
Gaussian score distribution, correct score sorting, non-divisible sample
batches, seeded sampling, a linear-map IVE case, abstention, preprocessing,
manifest selection, checkpoint wrappers, and published layer dimensions.
They do not validate learned DBCNN accuracy or the paper's runtime claims.
Full image inference and reproduction of the main tables require the missing
trained FS-IQA checkpoints and original experiment splits.

API references: [PyTorch installation](https://pytorch.org/get-started/locally/),
[torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html),
and [SciPy binomial distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html).
