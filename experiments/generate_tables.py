"""
Generate paper-ready CSV tables from reproducible experiment outputs.
"""

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd


TABLE1_COLUMNS = [
    "model_type",
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "kalman_enabled",
    "llm_agent_enabled",
    "anomaly_detector_enabled",
    "anomaly_detector_mode",
]

TABLE2_COLUMNS = [
    "config_name",
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "kalman_enabled",
    "anomaly_enabled",
    "llm_enabled",
    "model_type",
]

TABLE3_REQUIRED_COLUMNS = ["total_decisions"]
TABLE3_EXCLUDE_COLUMNS = {"config_name", "total_decisions"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready CSV tables.")
    parser.add_argument("--results-dir", default="experiments/results_phase2", help="Directory with experiment output CSV files")
    parser.add_argument("--user-study-dir", default="experiments/user_study_results", help="Directory with optional user-study CSV files")
    parser.add_argument("--output-dir", default="experiments/paper_tables", help="Directory to save generated paper tables")
    parser.add_argument("--strict", action="store_true", help="Fail if required table inputs are missing")
    return parser.parse_args()


def _missing_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> List[str]:
    return [col for col in required_columns if col not in df.columns]


def _select_existing_columns(df: pd.DataFrame, preferred_columns: Sequence[str]) -> List[str]:
    return [col for col in preferred_columns if col in df.columns]


def generate_tables(
    results_dir: Path,
    user_study_dir: Path,
    output_dir: Path,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[str] = []
    skipped: List[str] = []

    # Table 1: model metrics
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        skipped.append(f"table1_model_metrics.csv skipped: missing input file {results_csv}")
    else:
        df = pd.read_csv(results_csv)
        required = ["MAE", "MSE", "RMSE", "R2"]
        missing_required = _missing_columns(df, required)
        if missing_required:
            skipped.append(
                "table1_model_metrics.csv skipped: missing required columns in results.csv -> "
                + ", ".join(missing_required)
            )
        else:
            columns = _select_existing_columns(df, TABLE1_COLUMNS)
            table1 = df[columns].copy()
            out_path = output_dir / "table1_model_metrics.csv"
            table1.to_csv(out_path, index=False)
            generated.append(str(out_path))
            print(f"Generated table1_model_metrics.csv at {out_path}")

    # Table 2: ablation study
    ablation_csv = results_dir / "ablation_results.csv"
    if not ablation_csv.exists():
        skipped.append(f"table2_ablation_study.csv skipped: missing input file {ablation_csv}")
    else:
        df = pd.read_csv(ablation_csv)
        required = ["config_name", "MAE", "MSE", "RMSE", "R2"]
        missing_required = _missing_columns(df, required)
        if missing_required:
            skipped.append(
                "table2_ablation_study.csv skipped: missing required columns in ablation_results.csv -> "
                + ", ".join(missing_required)
            )
        else:
            columns = _select_existing_columns(df, TABLE2_COLUMNS)
            table2 = df[columns].copy()
            out_path = output_dir / "table2_ablation_study.csv"
            table2.to_csv(out_path, index=False)
            generated.append(str(out_path))
            print(f"Generated table2_ablation_study.csv at {out_path}")

    # Table 3: llm decision summary
    llm_csv = results_dir / "llm_decision_summary.csv"
    if not llm_csv.exists():
        skipped.append(f"table3_llm_decision_summary.csv skipped: missing input file {llm_csv}")
    else:
        df = pd.read_csv(llm_csv)
        missing_required = _missing_columns(df, TABLE3_REQUIRED_COLUMNS)
        if missing_required:
            skipped.append(
                "table3_llm_decision_summary.csv skipped: missing required columns in llm_decision_summary.csv -> "
                + ", ".join(missing_required)
            )
        else:
            action_columns = [c for c in df.columns if c not in TABLE3_EXCLUDE_COLUMNS]
            ordered = []
            if "config_name" in df.columns:
                ordered.append("config_name")
            ordered.extend(action_columns)
            ordered.append("total_decisions")
            table3 = df[ordered].copy()
            out_path = output_dir / "table3_llm_decision_summary.csv"
            table3.to_csv(out_path, index=False)
            generated.append(str(out_path))
            print(f"Generated table3_llm_decision_summary.csv at {out_path}")

    # Table 4: optional user study condition comparison
    condition_comparison_csv = user_study_dir / "condition_comparison.csv"
    if not condition_comparison_csv.exists():
        skipped.append(
            "table4_user_study_condition_comparison.csv skipped: missing optional input file "
            f"{condition_comparison_csv}"
        )
    else:
        table4 = pd.read_csv(condition_comparison_csv)
        out_path = output_dir / "table4_user_study_condition_comparison.csv"
        table4.to_csv(out_path, index=False)
        generated.append(str(out_path))
        print(f"Generated table4_user_study_condition_comparison.csv at {out_path}")

    if skipped:
        print("\nSkipped tables:")
        for item in skipped:
            print(f"- {item}")

    if strict and skipped:
        raise ValueError("Strict mode enabled and one or more tables were skipped due to missing required data.")

    print(f"\nGenerated {len(generated)} table(s) in {output_dir}")
    return generated, skipped


def main() -> None:
    args = parse_args()
    generate_tables(
        results_dir=Path(args.results_dir),
        user_study_dir=Path(args.user_study_dir),
        output_dir=Path(args.output_dir),
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
