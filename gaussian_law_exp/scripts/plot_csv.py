#!/usr/bin/env python3
"""
plot_csv.py

Read a CSV file and plot one or more columns.

Examples
--------
1. Plot "loss" against row index:
    python plot_csv.py results.csv --y loss

2. Plot multiple columns:
    python plot_csv.py results.csv --y train_loss val_loss

3. Use a CSV column as the x-axis:
    python plot_csv.py results.csv --x epoch --y train_loss val_loss

4. Save the figure:
    python plot_csv.py results.csv --x epoch --y loss --save loss.png

5. List all available CSV columns:
    python plot_csv.py results.csv --list-columns
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a CSV file and plot selected columns."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--x",
        type=str,
        default=None,
        help="Column to use as the x-axis. If omitted, the row index is used.",
    )
    parser.add_argument(
        "--y",
        nargs="+",
        default=None,
        help="One or more columns to plot on the y-axis.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Optional x-axis label.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Optional y-axis label.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path for saving the plot, e.g. figure.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive plot window.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print available CSV columns and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

    df = pd.read_csv(args.csv_file)

    if args.list_columns:
        print("Available columns:")
        for column in df.columns:
            print(f"  - {column}")
        return

    if args.y is None:
        raise ValueError(
            "Please specify at least one y-axis column using --y.\n"
            "Use --list-columns to inspect available columns."
        )

    requested_columns = list(args.y)
    if args.x is not None:
        requested_columns.append(args.x)

    missing_columns = [c for c in requested_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Column(s) not found: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    if args.x is None:
        x = df.index
        default_xlabel = "Index"
    else:
        x = df[args.x]
        default_xlabel = args.x

    fig, ax = plt.subplots(figsize=(9, 5))

    for column in args.y:
        ax.plot(x, df[column], label=column)

    ax.set_xlabel(args.xlabel or default_xlabel)
    ax.set_ylabel(args.ylabel or (args.y[0] if len(args.y) == 1 else "Value"))

    if args.title:
        ax.set_title(args.title)

    if len(args.y) > 1:
        ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
