"""
Create an ethical user-study CSV template and analyze real intake data.
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "participant_id",
    "condition",
    "task_id",
    "completion_time_seconds",
    "error_count",
    "satisfaction_score_1_to_5",
]
OPTIONAL_COLUMNS = ["notes"]
ALL_ALLOWED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
ALLOWED_CONDITIONS = {"baseline", "adaptive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a user-study CSV template or analyze user-study CSV input."
    )
    parser.add_argument(
        "--input-csv",
        required=False,
        help="Path to a real user-study CSV to validate and summarize.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/user_study_results",
        help="Directory for generated template and summary outputs.",
    )
    parser.add_argument(
        "--demo-template",
        action="store_true",
        help="Create an empty template CSV with headers only.",
    )
    return parser.parse_args()


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    unexpected = [col for col in df.columns if col not in ALL_ALLOWED_COLUMNS]
    if unexpected:
        raise ValueError(
            "Input CSV has unexpected columns: "
            + ", ".join(unexpected)
            + ". Allowed columns are: "
            + ", ".join(ALL_ALLOWED_COLUMNS)
        )


def _validate_values(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    validated["condition"] = validated["condition"].astype(str).str.strip().str.lower()

    bad_condition = ~validated["condition"].isin(ALLOWED_CONDITIONS)
    if bad_condition.any():
        invalid_values = sorted(set(validated.loc[bad_condition, "condition"].tolist()))
        raise ValueError(
            "Invalid condition value(s): "
            + ", ".join(map(str, invalid_values))
            + ". Allowed: baseline, adaptive."
        )

    validated["completion_time_seconds"] = pd.to_numeric(
        validated["completion_time_seconds"], errors="coerce"
    )
    if validated["completion_time_seconds"].isna().any():
        raise ValueError("completion_time_seconds contains non-numeric values.")
    if (validated["completion_time_seconds"] <= 0).any():
        raise ValueError("completion_time_seconds must be positive.")

    validated["error_count"] = pd.to_numeric(validated["error_count"], errors="coerce")
    if validated["error_count"].isna().any():
        raise ValueError("error_count contains non-numeric values.")
    if (validated["error_count"] < 0).any():
        raise ValueError("error_count must be >= 0.")

    validated["satisfaction_score_1_to_5"] = pd.to_numeric(
        validated["satisfaction_score_1_to_5"], errors="coerce"
    )
    if validated["satisfaction_score_1_to_5"].isna().any():
        raise ValueError("satisfaction_score_1_to_5 contains non-numeric values.")
    if (
        (validated["satisfaction_score_1_to_5"] < 1)
        | (validated["satisfaction_score_1_to_5"] > 5)
    ).any():
        raise ValueError("satisfaction_score_1_to_5 must be between 1 and 5.")

    return validated


def create_demo_template(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "user_study_template.csv"
    pd.DataFrame(columns=ALL_ALLOWED_COLUMNS).to_csv(template_path, index=False)
    return template_path


def analyze_input_csv(input_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    _validate_schema(df)
    validated = _validate_values(df)

    summary = pd.DataFrame(
        [
            {
                "total_rows": int(len(validated)),
                "unique_participants": int(validated["participant_id"].nunique()),
                "unique_tasks": int(validated["task_id"].nunique()),
                "mean_completion_time_seconds": float(
                    validated["completion_time_seconds"].mean()
                ),
                "mean_error_count": float(validated["error_count"].mean()),
                "mean_satisfaction_score_1_to_5": float(
                    validated["satisfaction_score_1_to_5"].mean()
                ),
            }
        ]
    )

    comparison = (
        validated.groupby("condition", as_index=False)
        .agg(
            count=("participant_id", "count"),
            mean_completion_time=("completion_time_seconds", "mean"),
            mean_error_count=("error_count", "mean"),
            mean_satisfaction_score=("satisfaction_score_1_to_5", "mean"),
        )
        .sort_values("condition")
    )

    summary_path = output_dir / "user_study_summary.csv"
    comparison_path = output_dir / "condition_comparison.csv"
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    return summary_path, comparison_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if not args.demo_template and not args.input_csv:
        raise ValueError("Provide either --demo-template, --input-csv, or both.")

    if args.demo_template:
        template_path = create_demo_template(output_dir)
        print(f"Saved template: {template_path}")

    if args.input_csv:
        summary_path, comparison_path = analyze_input_csv(Path(args.input_csv), output_dir)
        print(f"Saved: {summary_path}")
        print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
