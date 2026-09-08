#!/usr/bin/env python3
"""Compare DeepSets seed runs with the deterministic kernel-regression baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = {
    "kernel_regression": ("Kernel Regression", (0,)),
    "isi_context_deepsets": ("Neural Features Model", tuple(range(5))),
}
METRICS = {
    "NLL": "observation_nll",
    "Hellinger": "hellinger_distance_against_empirical",
    "KL": "kl_divergence_against_empirical",
    "W2_finite": "finite_renormalized_w2_against_empirical",
    "Tail": "tail_bin_error_against_empirical",
}


def aggregate(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {"models": {}}
    for model_name, (display_name, seeds) in MODELS.items():
        runs = []
        for seed in seeds:
            path = root / f"{model_name}_seed{seed}" / "results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics_by_split"]["test"]["observable_metrics"]
            runs.append({label: float(metrics[key]) for label, key in METRICS.items()})
        summary = {}
        for label in METRICS:
            values = [run[label] for run in runs]
            summary[label] = {
                "mean": statistics.fmean(values),
                "standard_deviation": statistics.pstdev(values),
                "values": values,
            }
        output["models"][model_name] = {
            "display_name": display_name,
            "seeds": list(seeds),
            "runs": runs,
            "summary": summary,
        }
    return output


def latex_number(value: float) -> str:
    if value == 0.0:
        return "0.00000"
    if abs(value) < 1.0e-4:
        exponent = f"{value:.2e}".split("e")
        return rf"{float(exponent[0]):g}\times10^{{{int(exponent[1])}}}"
    return f"{value:.5f}"


def latex_rows(payload: dict[str, Any]) -> str:
    lines = []
    for model_name in MODELS:
        model = payload["models"][model_name]
        cells = []
        for label in METRICS:
            summary = model["summary"][label]
            cells.append(
                rf"${latex_number(summary['mean'])} \pm {latex_number(summary['standard_deviation'])}$"
            )
        lines.append(f"{model['display_name']} & " + " & ".join(cells) + r"\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/table3_matrix"))
    args = parser.parse_args()
    payload = aggregate(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = latex_rows(payload)
    (args.root / "table3_rows.tex").write_text(rows, encoding="utf-8")
    print(rows, end="")


if __name__ == "__main__":
    main()
