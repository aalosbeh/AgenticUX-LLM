"""
Generate figures from experiment outputs without fabricating data.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures from experiment outputs.")
    parser.add_argument("--results-dir", default="experiments/results", help="Directory with results.csv and predictions.csv")
    parser.add_argument("--output-dir", default="experiments/figures", help="Directory to save generated figures")
    parser.add_argument("--strict", action="store_true", help="Fail if any configured figure cannot be generated")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> List[str]:
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def synthetic_tag(predictions_df: pd.DataFrame) -> str:
    if "is_synthetic_demo" in predictions_df.columns and predictions_df["is_synthetic_demo"].astype(bool).all():
        return " (synthetic demo)"
    return ""


def fig_metrics_bar(results_df: pd.DataFrame, _: pd.DataFrame) -> plt.Figure:
    metrics = ["MAE", "MSE", "RMSE", "R2"]
    values = [float(results_df[col].iloc[0]) for col in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, values, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.set_title("Evaluation Metrics")
    ax.set_ylabel("Value")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    return fig


def fig_actual_vs_predicted(_: pd.DataFrame, predictions_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    x = predictions_df["actual_cognitive_load"].to_numpy()
    y = predictions_df["predicted_cognitive_load"].to_numpy()
    ax.scatter(x, y, alpha=0.4, s=14)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual cognitive load")
    ax.set_ylabel("Predicted cognitive load")
    ax.set_title(f"Actual vs Predicted{synthetic_tag(predictions_df)}")
    return fig


def fig_residual_hist(_: pd.DataFrame, predictions_df: pd.DataFrame) -> plt.Figure:
    residuals = predictions_df["actual_cognitive_load"] - predictions_df["predicted_cognitive_load"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, alpha=0.8, color="#2196F3", edgecolor="black")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution{synthetic_tag(predictions_df)}")
    return fig


def fig_load_level_counts(_: pd.DataFrame, predictions_df: pd.DataFrame) -> plt.Figure:
    counts = predictions_df["load_level"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#4CAF50")
    ax.set_xlabel("Predicted load level")
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted Load Level Distribution{synthetic_tag(predictions_df)}")
    return fig


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "results.csv"
    predictions_path = results_dir / "predictions.csv"
    dataframes: Dict[str, pd.DataFrame] = {}
    if results_path.exists():
        dataframes["results"] = pd.read_csv(results_path)
    if predictions_path.exists():
        dataframes["predictions"] = pd.read_csv(predictions_path)

    figure_specs: List[Dict[str, object]] = [
        {
            "name": "fig_metrics_bar",
            "filename": "fig_metrics_bar",
            "required_df": "results",
            "required_columns": ["MAE", "MSE", "RMSE", "R2"],
            "builder": fig_metrics_bar,
        },
        {
            "name": "fig_actual_vs_predicted",
            "filename": "fig_actual_vs_predicted",
            "required_df": "predictions",
            "required_columns": ["actual_cognitive_load", "predicted_cognitive_load"],
            "builder": fig_actual_vs_predicted,
        },
        {
            "name": "fig_residual_distribution",
            "filename": "fig_residual_distribution",
            "required_df": "predictions",
            "required_columns": ["actual_cognitive_load", "predicted_cognitive_load"],
            "builder": fig_residual_hist,
        },
        {
            "name": "fig_load_level_counts",
            "filename": "fig_load_level_counts",
            "required_df": "predictions",
            "required_columns": ["load_level"],
            "builder": fig_load_level_counts,
        },
        {"name": "fig1_nasa_tlx_comparison", "legacy_missing": "table1_summary_statistics.csv"},
        {"name": "fig2_nasa_tlx_components", "legacy_missing": "table2_nasa_tlx_components.csv"},
        {"name": "fig3_completion_time", "legacy_missing": "table3_task_specific.csv"},
        {"name": "fig4_error_rates", "legacy_missing": "table1_summary_statistics.csv"},
        {"name": "fig5_navigation_efficiency", "legacy_missing": "table1_summary_statistics.csv"},
        {"name": "fig6_sus_scores", "legacy_missing": "table1_summary_statistics.csv"},
        {"name": "fig7_physiological_measures", "legacy_missing": "table1_summary_statistics.csv"},
        {"name": "fig8_task_specific_performance", "legacy_missing": "table3_task_specific.csv"},
        {"name": "fig9_system_performance", "legacy_missing": "table6_system_performance.csv"},
        {"name": "fig10_system_comparison", "legacy_missing": "table4_system_comparison.csv"},
        {"name": "fig11_demographic_analysis", "legacy_missing": "table5_demographic_analysis.csv"},
    ]

    generated: List[str] = []
    skipped: List[str] = []

    for spec in figure_specs:
        if "legacy_missing" in spec:
            msg = f"{spec['name']} skipped: requires {spec['legacy_missing']} (not available in results/predictions pipeline)"
            skipped.append(msg)
            continue

        df_key = str(spec["required_df"])
        required_columns = list(spec["required_columns"])
        if df_key not in dataframes:
            skipped.append(f"{spec['name']} skipped: missing {df_key}.csv in {results_dir}")
            continue

        missing_columns = [col for col in required_columns if col not in dataframes[df_key].columns]
        if missing_columns:
            skipped.append(f"{spec['name']} skipped: missing columns in {df_key}.csv -> {', '.join(missing_columns)}")
            continue

        builder: Callable[[pd.DataFrame, pd.DataFrame], plt.Figure] = spec["builder"]  # type: ignore[assignment]
        fig = builder(dataframes.get("results", pd.DataFrame()), dataframes.get("predictions", pd.DataFrame()))
        paths = save_figure(fig, output_dir, str(spec["filename"]))
        generated.extend(paths)
        print(f"Generated {spec['name']}: {paths[0]} and {paths[1]}")

    if skipped:
        print("\nSkipped figures:")
        for entry in skipped:
            print(f"- {entry}")

    if args.strict and skipped:
        raise ValueError("Strict mode enabled and some figures were skipped due to missing required data.")

    print(f"\nGenerated {len(generated) // 2} figures in {output_dir}")


if __name__ == "__main__":
    main()
