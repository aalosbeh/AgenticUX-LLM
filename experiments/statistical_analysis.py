"""
Minimal statistical summary for experiment result metrics.
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["MAE", "MSE", "RMSE", "R2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment results.csv.")
    parser.add_argument("--results-csv", required=True, help="Path to results.csv.")
    parser.add_argument("--output-dir", default="experiments/results", help="Directory for summary_stats.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(args.results_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in results_df.columns]
    if missing:
        raise ValueError(
            "results.csv is missing required metric columns: " + ", ".join(missing)
        )

    summary_rows = []
    for metric in REQUIRED_COLUMNS:
        series = pd.to_numeric(results_df[metric], errors="coerce")
        if series.isna().any():
            raise ValueError(f"Column '{metric}' contains non-numeric values.")
        summary_rows.append(
            {
                "metric": metric,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    output_path = output_dir / "summary_stats.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
