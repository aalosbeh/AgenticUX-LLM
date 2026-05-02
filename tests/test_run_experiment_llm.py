import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

import pandas as pd

from experiments.run_experiment import make_synthetic_demo, run_single_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "run_experiment.py"


class TestRunExperimentLLM(unittest.TestCase):
    def test_predictions_include_llm_columns_when_enabled(self):
        df = make_synthetic_demo(n_samples=80, seed=42)
        run_data = run_single_configuration(
            df=df,
            data_source="synthetic_demo",
            seed=42,
            model_type="auto",
            use_kalman=True,
            use_anomaly_detector=True,
            use_llm_agent=True,
        )
        predictions_df = run_data["predictions_df"]
        self.assertIn("llm_action", predictions_df.columns)
        self.assertIn("llm_reasoning", predictions_df.columns)
        self.assertIn("llm_confidence", predictions_df.columns)
        self.assertIn("llm_mode", predictions_df.columns)

    def test_ablation_with_llm_writes_decision_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(RUN_EXPERIMENT_SCRIPT),
                "--synthetic-demo",
                "--n-samples",
                "100",
                "--seed",
                "42",
                "--output-dir",
                tmpdir,
                "--run-ablation",
            ]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            output_path = Path(tmpdir) / "llm_decision_summary.csv"
            self.assertTrue(output_path.exists())
            loaded = pd.read_csv(output_path)
            self.assertIn("simplify_ui", loaded.columns)
            self.assertIn("highlight_relevant", loaded.columns)
            self.assertIn("no_change", loaded.columns)
            self.assertIn("request_human_review", loaded.columns)

    def test_single_run_with_llm_writes_decision_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(RUN_EXPERIMENT_SCRIPT),
                "--synthetic-demo",
                "--n-samples",
                "100",
                "--seed",
                "42",
                "--output-dir",
                tmpdir,
                "--use-llm-agent",
            ]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            output_path = Path(tmpdir) / "llm_decision_summary.csv"
            self.assertTrue(output_path.exists())
            loaded = pd.read_csv(output_path)
            self.assertEqual(len(loaded), 1)
            self.assertIn("config_name", loaded.columns)
            self.assertEqual(str(loaded.iloc[0]["config_name"]), "single_run")
            self.assertIn("simplify_ui", loaded.columns)
            self.assertIn("highlight_relevant", loaded.columns)
            self.assertIn("no_change", loaded.columns)
            self.assertIn("request_human_review", loaded.columns)
            self.assertIn("total_decisions", loaded.columns)


if __name__ == "__main__":
    unittest.main()
