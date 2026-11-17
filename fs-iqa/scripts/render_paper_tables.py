#!/usr/bin/env python3
"""Render the main paper's archived comparison CSV as Markdown tables."""

import csv
from pathlib import Path


def render_tables(csv_path):
    with Path(csv_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = ["RS-Reg", "Cert-Reg", "MS", "DMS", "DMS-IQA", "FS-IQA"]
    text = []
    for task, models, dataset, table in (
        ("NR", ["KonCept", "DBCNN", "Linearity"], "KonIQ-10k", 1),
        ("FR", ["LPIPS", "DISTS", "DreamSim"], "KADID-10k", 2),
    ):
        text.append(f"### {task}-IQA · {dataset}\n")
        for regime, sigma in (("weak", "0.1"), ("strong", "1.0")):
            subset = [r for r in rows if r["task"] == task and r["regime"] == regime]
            text.append(f"**{regime.capitalize()} setting** (FS-IQA σf = {sigma}). "
                        f"Paper Table {table}. Each cell: **SRC ↑ / εy ↓**.\n")
            headings = []
            for model in models:
                radius = next(r["target_mean_epsilon_x"] for r in subset if r["backbone"] == model)
                headings.append(f"{model}<br>εx = {radius}")
            text.append("| Method | " + " | ".join(headings) + " |")
            text.append("|:--|--:|--:|--:|")
            for method in methods:
                cells = []
                for model in models:
                    group = [r for r in subset if r["backbone"] == model]
                    record = next(r for r in group if r["method"] == method)
                    best_src = max(float(r["src"]) for r in group)
                    best_eps = min(float(r["epsilon_y"]) for r in group)
                    def fmt(value, best):
                        return f"**{value}**" if float(value) == best else value
                    cells.append(fmt(record["src"], best_src) + " / " + fmt(record["epsilon_y"], best_eps))
                label = "**FS-IQA**" if method == "FS-IQA" else method
                text.append("| " + label + " | " + " | ".join(cells) + " |")
            text.append("")
    return "\n".join(text)


if __name__ == "__main__":
    print(render_tables(Path(__file__).resolve().parents[1] / "results/paper_comparison.csv"))
