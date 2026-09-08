#!/usr/bin/env python3
"""Aggregate five-seed test metrics as mean and population standard deviation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "global_constant": "Empirical Parameter Estimation",
    "pathwise_mlp": "Pointwise MLP",
    "oracle_feature": "Fixed Features Model",
    "cylindrical": "Neural Features Model",
}
METRICS = {
    "nll": ("observable_metrics", "nll"),
    "e_m": ("synthetic_diagnostic_metrics", "e_m"),
    "e_C": ("synthetic_diagnostic_metrics", "e_C"),
    "W_2": ("synthetic_diagnostic_metrics", "gaussian_w2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="cosine5")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    return parser.parse_args()


def format_standard_deviation(value: float) -> str:
    """Keep very small nonzero standard deviations visible in generated LaTeX."""
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-4:
        exponent = math.floor(math.log10(abs(value)))
        coefficient = value / (10.0**exponent)
        return f"{coefficient:.3f}{{\\times}}10^{{{exponent}}}"
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / "experiments"
    summary: dict[str, Any] = {
        "prefix": args.prefix,
        "seeds": args.seeds,
        "standard_deviation": "population",
        "models": {},
    }
    rows: list[dict[str, Any]] = []
    latex_rows: list[str] = []
    for model, display_name in MODELS.items():
        values = {metric: [] for metric in METRICS}
        for seed in args.seeds:
            result_path = output_root / f"{args.prefix}_{model}_seed{seed}" / "results.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            test_metrics = payload["metrics_by_split"]["test"]
            for metric, (group, key) in METRICS.items():
                values[metric].append(float(test_metrics[group][key]))
        aggregates = {
            metric: {
                "mean": statistics.fmean(metric_values),
                "standard_deviation": statistics.pstdev(metric_values),
                "values": metric_values,
            }
            for metric, metric_values in values.items()
        }
        summary["models"][model] = {"display_name": display_name, "metrics": aggregates}
        row = {"model": model, "display_name": display_name}
        for metric, aggregate in aggregates.items():
            row[f"{metric}_mean"] = aggregate["mean"]
            row[f"{metric}_std"] = aggregate["standard_deviation"]
        rows.append(row)
        cells = [
            f"${aggregates[metric]['mean']:.4f} \\pm "
            f"{format_standard_deviation(aggregates[metric]['standard_deviation'])}$"
            for metric in METRICS
        ]
        if model == "cylindrical":
            name_cell = f"\\textbf{{{display_name}}}"
            cells = [f"$\\boldsymbol{{{cell[1:-1]}}}$" for cell in cells]
        else:
            name_cell = display_name
        latex_rows.append(f"{name_cell} & " + " & ".join(cells) + r"\\")

    json_path = output_root / f"{args.prefix}_summary.json"
    csv_path = output_root / f"{args.prefix}_summary.csv"
    tex_path = output_root / f"{args.prefix}_table_rows.tex"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
