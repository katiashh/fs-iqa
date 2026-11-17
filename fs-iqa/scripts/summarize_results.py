#!/usr/bin/env python3
"""Summarize an exported results CSV without model weights or PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def summarize(path):
    frame = pd.read_csv(path)
    required = {"median", "lower_b", "upper_b", "gt"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    for column in required | ({"epsilon_x", "epsilon_x_estimate"} & set(frame)):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    paired = np.isfinite(frame["median"]) & np.isfinite(frame["gt"])
    rank_ok = (paired.sum() >= 2 and frame.loc[paired, "median"].nunique() > 1
               and frame.loc[paired, "gt"].nunique() > 1)
    bounds = np.isfinite(frame.lower_b) & np.isfinite(frame.upper_b)
    radius_column = next((c for c in ("epsilon_x_estimate", "epsilon_x") if c in frame), None)
    radii = frame[radius_column] if radius_column else pd.Series(dtype=float)
    radii = radii[np.isfinite(radii)]
    return {
        "rows": len(frame),
        "valid_score_mos_pairs": int(paired.sum()),
        "src": float(spearmanr(frame.loc[paired, "median"], frame.loc[paired, "gt"]).statistic)
        if rank_ok else None,
        "mean_reported_interval_width": float((frame.upper_b - frame.lower_b)[bounds].mean())
        if bounds.any() else None,
        "mean_reported_input_radius": float(radii.mean()) if len(radii) else None,
        "finite_interval_rows": int(bounds.sum()),
        "reversed_interval_rows": int((bounds & (frame.lower_b > frame.upper_b)).sum()),
        "median_outside_reported_interval_rows": int((bounds & np.isfinite(frame["median"]) & (
            (frame["median"] < frame.lower_b) | (frame["median"] > frame.upper_b))).sum()),
        "source": str(path),
        "note": "Descriptive statistics of supplied values; this does not validate certificates.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", type=Path, default=Path("results/results_on_koniq_subset.csv"))
    p.add_argument("--json", type=Path, help="Optionally save the summary")
    args = p.parse_args()
    try:
        result = summarize(args.csv)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(output, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output)


if __name__ == "__main__":
    main()
