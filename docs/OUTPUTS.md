# Output files

Every cleaned CLI run writes a CSV and a JSON sidecar with the same basename.
The CSV is flushed after each image. JSON `run_status` is `complete` only after
the entire selected list finishes; `completed_images` records the count. A
hard termination may leave `run_status=running`, so incomplete runs are not
silently reported as complete.

## CSV schema

| Column | Meaning |
|---|---|
| `image_name` | Image path relative to `--image-dir` |
| `median` | Monte Carlo median quality score |
| `lower_b`, `upper_b` | Finite-sample bounds in feature space; blank if unavailable or not requested |
| `gt` | MOS in the original units of the selected input column; blank if no MOS CSV |
| `epsilon_x_estimate` | Empirical IVE input-space L2 radius; not a worst-case certificate |
| `jacobian_norm` | Spectral norm of the local feature Jacobian |
| `curvature_proxy` | Maximum sampled directional curvature of the sum of feature outputs |
| `interval_status` | `finite`, `abstained`, or `not_requested` |
| `ive_status` | `empirical_proxy`, `abstained`, or `not_requested` |
| `n_samples` | Actual number of feature-noise samples |
| `seconds` | Per-image model computation time, including the requested IVE; excludes image decoding and model loading |

The output score is not clamped to `[0, 1]`; the final scorer is linear. The
paper reports normalized-score results, but clipping features does not enforce
a bounded scorer output.

## JSON metadata

The sidecar records CLI arguments, the feature-space radius `kappa * sigma`,
the software versions, selected device, exact selected image list and its
SHA-256, and completion status. It records checkpoint paths, not checkpoint
hashes; keep the corresponding weights when preserving an experiment.

## Original results export

`results/results_on_koniq_subset.csv` is unchanged. Its seven original columns
are `median`, `lower_b`, `upper_b`, `gt`, `epsilon_x`, `L_emp`, and `vol_bound`.
They are archival outputs, not results of the cleaned CLI. The names
`epsilon_x`, `L_emp`, and `vol_bound` do not establish deterministic input-space
guarantees; see [implementation notes](IMPLEMENTATION.md).

Summarize either schema with:

```bash
python scripts/summarize_results.py results/results_on_koniq_subset.csv
python scripts/summarize_results.py outputs/dbcnn_ive.csv --json outputs/summary.json
```

This command reports SRC, mean recorded interval width, mean recorded radius,
and basic interval consistency counts. It does not infer missing run settings
or validate a robustness certificate.
