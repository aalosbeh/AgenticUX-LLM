"""
Deterministic experiment runner for the implemented cognitive load model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.cognitive_load_model import CognitiveLoadInput, CognitiveLoadModel
from src.core.anomaly_detector import BehaviorAnomalyDetector
from src.agents.llm_agent import LLMAgent


REQUIRED_FEATURE_COLUMNS = [
    "mouse_velocity",
    "click_frequency",
    "time_between_actions",
    "error_count",
    "correction_count",
    "page_visits",
    "heart_rate",
    "pupil_dilation",
    "blink_rate",
    "task_complexity",
    "task_familiarity",
    "time_pressure",
    "element_density",
    "color_complexity",
]
TARGET_COLUMN = "cognitive_load"
ABLATION_CONFIGS = [
    {
        "config_name": "baseline (residual_linear)",
        "model_type": "residual_linear",
        "use_kalman": False,
        "use_anomaly_detector": False,
        "use_llm_agent": False,
    },
    {
        "config_name": "real_gb",
        "model_type": "real_gb",
        "use_kalman": False,
        "use_anomaly_detector": False,
        "use_llm_agent": False,
    },
    {
        "config_name": "+kalman",
        "model_type": "real_gb",
        "use_kalman": True,
        "use_anomaly_detector": False,
        "use_llm_agent": False,
    },
    {
        "config_name": "+anomaly",
        "model_type": "real_gb",
        "use_kalman": True,
        "use_anomaly_detector": True,
        "use_llm_agent": False,
    },
    {
        "config_name": "+llm",
        "model_type": "real_gb",
        "use_kalman": True,
        "use_anomaly_detector": True,
        "use_llm_agent": True,
    },
]
LLM_ACTIONS = [
    "simplify_ui",
    "highlight_relevant",
    "no_change",
    "request_human_review",
]
LLM_FULL_PASS_MAX_ROWS = 2000


def build_llm_decision_summary(predictions_df: pd.DataFrame, config_name: str = "single_run") -> pd.DataFrame:
    action_counts = (
        predictions_df["llm_action"].value_counts().to_dict()
        if "llm_action" in predictions_df.columns
        else {}
    )
    return pd.DataFrame(
        [
            {
                "config_name": config_name if config_name else "single_run",
                "simplify_ui": int(action_counts.get("simplify_ui", 0)),
                "highlight_relevant": int(action_counts.get("highlight_relevant", 0)),
                "no_change": int(action_counts.get("no_change", 0)),
                "request_human_review": int(action_counts.get("request_human_review", 0)),
                "total_decisions": int(sum(action_counts.get(a, 0) for a in LLM_ACTIONS)),
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic cognitive load experiment.")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input CSV data.")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Output directory.")
    parser.add_argument("--synthetic-demo", action="store_true", help="Use synthetic demo data.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Synthetic sample count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs.")
    parser.add_argument("--use-llm-agent", action="store_true", help="Enable optional LLM agent.")
    parser.add_argument("--use-kalman", action="store_true", help="Enable Kalman smoothing in model output.")
    parser.add_argument(
        "--use-anomaly-detector",
        action="store_true",
        help="Enable optional anomaly scoring.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "real_gb", "residual_linear"],
        help="Gradient boosting backend selection.",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run predefined ablation configurations and save ablation CSVs.",
    )
    return parser.parse_args()


def make_synthetic_demo(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "mouse_velocity": rng.normal(260, 80, n_samples).clip(30, 900),
            "click_frequency": rng.normal(2.0, 0.7, n_samples).clip(0.1, 5.0),
            "time_between_actions": rng.normal(1.1, 0.5, n_samples).clip(0.1, 5.0),
            "error_count": rng.poisson(1.2, n_samples).clip(0, 10),
            "correction_count": rng.poisson(0.8, n_samples).clip(0, 10),
            "page_visits": rng.integers(1, 10, n_samples),
            "heart_rate": rng.normal(80, 12, n_samples).clip(45, 180),
            "pupil_dilation": rng.normal(4.0, 0.8, n_samples).clip(2.0, 8.0),
            "blink_rate": rng.normal(18, 5, n_samples).clip(5, 35),
            "task_complexity": rng.uniform(0.0, 1.0, n_samples),
            "task_familiarity": rng.uniform(0.0, 1.0, n_samples),
            "time_pressure": rng.uniform(0.0, 1.0, n_samples),
            "element_density": rng.uniform(0.0, 1.0, n_samples),
            "color_complexity": rng.uniform(0.0, 1.0, n_samples),
        }
    )

    # Synthetic label with transparent construction for demo purposes only.
    norm = pd.DataFrame(
        {
            "mouse_velocity": (df["mouse_velocity"] / 1000).clip(0, 1),
            "click_frequency": (df["click_frequency"] / 5).clip(0, 1),
            "time_between_actions": (df["time_between_actions"] / 5).clip(0, 1),
            "error_count": (df["error_count"] / 10).clip(0, 1),
            "correction_count": (df["correction_count"] / 10).clip(0, 1),
            "page_visits": (df["page_visits"] / 10).clip(0, 1),
            "heart_rate": (df["heart_rate"] / 200).clip(0, 1),
            "pupil_dilation": (df["pupil_dilation"] / 8).clip(0, 1),
            "blink_rate": (df["blink_rate"] / 30).clip(0, 1),
            "task_complexity": df["task_complexity"],
            "task_familiarity": df["task_familiarity"],
            "time_pressure": df["time_pressure"],
            "ui_complexity": ((df["element_density"] + df["color_complexity"]) / 2).clip(0, 1),
        }
    )
    base_signal = (
        0.12 * norm["mouse_velocity"]
        + 0.10 * norm["click_frequency"]
        + 0.08 * (1.0 - norm["time_between_actions"])
        + 0.12 * norm["error_count"]
        + 0.08 * norm["correction_count"]
        + 0.05 * norm["page_visits"]
        + 0.05 * norm["heart_rate"]
        + 0.04 * norm["pupil_dilation"]
        + 0.03 * norm["blink_rate"]
        + 0.14 * norm["task_complexity"]
        + 0.10 * (1.0 - norm["task_familiarity"])
        + 0.06 * norm["time_pressure"]
        + 0.03 * norm["ui_complexity"]
    )
    noise = rng.normal(0, 0.04, n_samples)
    df[TARGET_COLUMN] = (100 * (base_signal + noise)).clip(0, 100)
    df["data_source"] = "synthetic_demo"
    return df


def load_input_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        missing = [c for c in REQUIRED_FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )
        return df.copy(), "input_csv"

    if args.synthetic_demo:
        return make_synthetic_demo(args.n_samples, args.seed), "synthetic_demo"

    raise ValueError("Provide either --input-csv or --synthetic-demo.")


def row_to_model_input(row: pd.Series) -> CognitiveLoadInput:
    return CognitiveLoadInput(
        mouse_velocity=float(row["mouse_velocity"]),
        click_frequency=float(row["click_frequency"]),
        time_between_actions=float(row["time_between_actions"]),
        error_count=int(row["error_count"]),
        correction_count=int(row["correction_count"]),
        page_visits=int(row["page_visits"]),
        heart_rate=float(row["heart_rate"]),
        pupil_dilation=float(row["pupil_dilation"]),
        blink_rate=float(row["blink_rate"]),
        task_complexity=float(row["task_complexity"]),
        task_familiarity=float(row["task_familiarity"]),
        time_pressure=float(row["time_pressure"]),
        element_density=float(row["element_density"]),
        color_complexity=float(row["color_complexity"]),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(np.square(errors)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def run_single_configuration(
    *,
    df: pd.DataFrame,
    data_source: str,
    seed: int,
    model_type: str,
    use_kalman: bool,
    use_anomaly_detector: bool,
    use_llm_agent: bool,
) -> Dict[str, Any]:
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("Need at least 2 rows to run train/evaluation split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    split_idx = max(1, int(0.8 * n_rows))
    if split_idx >= n_rows:
        split_idx = n_rows - 1
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    train_data: List[Tuple[CognitiveLoadInput, float]] = []
    for _, row in df.iloc[train_idx].iterrows():
        train_data.append((row_to_model_input(row), float(row[TARGET_COLUMN])))

    model = CognitiveLoadModel(model_type=model_type, use_kalman=use_kalman)
    model.train(train_data)
    stats = model.get_model_stats()

    llm_agent = LLMAgent() if use_llm_agent else None
    anomaly_scores = np.full(n_rows, np.nan, dtype=float)
    anomaly_enabled = False
    anomaly_mode = "disabled"
    if use_anomaly_detector:
        feature_matrix = df[REQUIRED_FEATURE_COLUMNS].to_numpy(dtype=float)
        try:
            anomaly_detector = BehaviorAnomalyDetector()
            anomaly_detector.fit(feature_matrix[train_idx])
            anomaly_scores = anomaly_detector.score(feature_matrix)
            anomaly_enabled = True
            anomaly_mode = "isolation_forest"
        except ImportError:
            anomaly_detector = BehaviorAnomalyDetector(fallback_mode="zscore")
            anomaly_detector.fit(feature_matrix[train_idx])
            anomaly_scores = anomaly_detector.score(feature_matrix)
            anomaly_enabled = True
            anomaly_mode = "zscore_fallback"

    records = []
    y_true = []
    y_pred = []
    split_labels = np.array(["train"] * n_rows, dtype=object)
    split_labels[test_idx] = "test"
    llm_eval_indices = set()
    if llm_agent is not None:
        if n_rows <= LLM_FULL_PASS_MAX_ROWS:
            llm_eval_indices = set(range(n_rows))
        else:
            sampled = rng.choice(
                np.arange(n_rows),
                size=LLM_FULL_PASS_MAX_ROWS,
                replace=False,
            )
            llm_eval_indices = set(int(i) for i in sampled.tolist())

    for row_idx, row in df.iterrows():
        pred = model.predict(row_to_model_input(row))
        actual = float(row[TARGET_COLUMN])
        predicted = float(pred["cognitive_load"])
        residual = actual - predicted
        record = {
            "row_index": int(row_idx),
            "split": split_labels[row_idx],
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
            "actual_cognitive_load": actual,
            "predicted_cognitive_load": predicted,
            "load_level": pred["load_level"],
            "confidence": float(pred["confidence"]),
            "data_source": data_source,
            "is_synthetic_demo": bool(data_source == "synthetic_demo"),
            "model_type": stats["gb_model_type"],
            "kalman_enabled": bool(stats["kalman_enabled"]),
        }
        if anomaly_enabled:
            record["anomaly_score"] = float(anomaly_scores[row_idx])
        if llm_agent is not None:
            if int(row_idx) in llm_eval_indices:
                llm_features = {
                    **row.to_dict(),
                    "predicted_cognitive_load": predicted,
                    "actual_cognitive_load": actual,
                    "anomaly_score": float(anomaly_scores[row_idx]) if anomaly_enabled else 0.0,
                }
                llm_result = llm_agent.analyze_user_state(llm_features)
                record["llm_action"] = llm_result["action"]
                record["llm_reasoning"] = llm_result["reasoning"]
                record["llm_confidence"] = float(llm_result["confidence"])
                record["llm_mode"] = llm_result["mode"]
            else:
                record["llm_action"] = "no_change"
                record["llm_reasoning"] = "Deterministic sampled subset: row not evaluated by LLM."
                record["llm_confidence"] = 0.0
                record["llm_mode"] = "sampled_subset"
        records.append({**record})
        if split_labels[row_idx] == "test":
            y_true.append(actual)
            y_pred.append(predicted)

    metrics = regression_metrics(np.array(y_true), np.array(y_pred))
    results_row = {
        "data_source": data_source,
        "is_synthetic_demo": bool(data_source == "synthetic_demo"),
        "seed": int(seed),
        "n_samples_total": int(n_rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "model_type": stats["gb_model_type"],
        "kalman_enabled": bool(stats["kalman_enabled"]),
        "llm_agent_enabled": bool(use_llm_agent),
        "anomaly_detector_enabled": bool(anomaly_enabled),
        "anomaly_detector_mode": anomaly_mode,
        **metrics,
    }

    config = {
        "input_csv": None,
        "synthetic_demo": bool(data_source == "synthetic_demo"),
        "seed": int(seed),
        "use_llm_agent": bool(use_llm_agent),
        "use_kalman": bool(use_kalman),
        "use_anomaly_detector": bool(use_anomaly_detector),
        "model_type_requested": model_type,
        "model_type_resolved": stats["gb_model_type"],
        "kalman_enabled": bool(stats["kalman_enabled"]),
        "llm_mode": "enabled" if use_llm_agent else "disabled",
        "anomaly_detector_enabled": bool(anomaly_enabled),
        "anomaly_detector_mode": anomaly_mode,
        "resolved_data_source": data_source,
        "model": "CognitiveLoadModel",
    }

    return {
        "results_row": results_row,
        "predictions_df": pd.DataFrame(records),
        "config": config,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, data_source = load_input_dataframe(args)
    if args.run_ablation:
        ablation_rows = []
        llm_summary_rows = []
        for cfg in ABLATION_CONFIGS:
            run_data = run_single_configuration(
                df=df,
                data_source=data_source,
                seed=args.seed,
                model_type=cfg["model_type"],
                use_kalman=cfg["use_kalman"],
                use_anomaly_detector=cfg["use_anomaly_detector"],
                use_llm_agent=cfg["use_llm_agent"],
            )
            row = {
                "config_name": cfg["config_name"],
                "MAE": run_data["results_row"]["MAE"],
                "MSE": run_data["results_row"]["MSE"],
                "RMSE": run_data["results_row"]["RMSE"],
                "R2": run_data["results_row"]["R2"],
                "kalman_enabled": bool(run_data["results_row"]["kalman_enabled"]),
                "anomaly_enabled": bool(run_data["results_row"]["anomaly_detector_enabled"]),
                "llm_enabled": bool(run_data["results_row"]["llm_agent_enabled"]),
                "model_type": run_data["results_row"]["model_type"],
            }
            ablation_rows.append(row)
            if cfg["use_llm_agent"]:
                summary_row = build_llm_decision_summary(
                    run_data["predictions_df"], config_name=cfg["config_name"]
                ).iloc[0].to_dict()
                llm_summary_rows.append(summary_row)

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_results_path = output_dir / "ablation_results.csv"
        ablation_summary_path = output_dir / "ablation_summary_table.csv"
        llm_decision_summary_path = output_dir / "llm_decision_summary.csv"
        ablation_df.to_csv(ablation_results_path, index=False)
        ablation_df.to_csv(ablation_summary_path, index=False)
        pd.DataFrame(llm_summary_rows).to_csv(llm_decision_summary_path, index=False)

        print(f"Saved: {ablation_results_path}")
        print(f"Saved: {ablation_summary_path}")
        print(f"Saved: {llm_decision_summary_path}")
        if data_source == "synthetic_demo":
            print("Note: Ablation results are from synthetic demo data only and are not human evidence.")
        return

    run_data = run_single_configuration(
        df=df,
        data_source=data_source,
        seed=args.seed,
        model_type=args.model_type,
        use_kalman=args.use_kalman,
        use_anomaly_detector=args.use_anomaly_detector,
        use_llm_agent=args.use_llm_agent,
    )
    results_df = pd.DataFrame([run_data["results_row"]])
    predictions_df = run_data["predictions_df"]

    results_path = output_dir / "results.csv"
    predictions_path = output_dir / "predictions.csv"
    config_path = output_dir / "config.json"
    llm_decision_summary_path = output_dir / "llm_decision_summary.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    if args.use_llm_agent:
        llm_summary_df = build_llm_decision_summary(predictions_df, config_name="single_run")
        llm_summary_df.to_csv(llm_decision_summary_path, index=False)
    config = {
        "input_csv": args.input_csv,
        "output_dir": str(output_dir),
        "synthetic_demo": bool(args.synthetic_demo),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "use_llm_agent": bool(args.use_llm_agent),
        "use_kalman": bool(args.use_kalman),
        "use_anomaly_detector": bool(args.use_anomaly_detector),
        "model_type_requested": args.model_type,
        "model_type_resolved": run_data["results_row"]["model_type"],
        "kalman_enabled": bool(run_data["results_row"]["kalman_enabled"]),
        "llm_mode": "enabled" if args.use_llm_agent else "disabled",
        "anomaly_detector_enabled": bool(run_data["results_row"]["anomaly_detector_enabled"]),
        "anomaly_detector_mode": run_data["results_row"]["anomaly_detector_mode"],
        "resolved_data_source": data_source,
        "model": "CognitiveLoadModel",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Saved: {results_path}")
    print(f"Saved: {predictions_path}")
    if args.use_llm_agent:
        print(f"Saved: {llm_decision_summary_path}")
    print(f"Saved: {config_path}")
    if data_source == "synthetic_demo":
        print("Note: Results are from synthetic demo data only and are not human evidence.")


if __name__ == "__main__":
    main()
