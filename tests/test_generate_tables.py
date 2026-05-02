import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.generate_tables import generate_tables


class TestGenerateTables(unittest.TestCase):
    def _write_core_inputs(self, results_dir: Path) -> None:
        pd.DataFrame(
            [
                {
                    "model_type": "real_gb",
                    "MAE": 1.0,
                    "MSE": 2.0,
                    "RMSE": 1.414,
                    "R2": 0.8,
                    "kalman_enabled": True,
                    "llm_agent_enabled": True,
                    "anomaly_detector_enabled": True,
                    "anomaly_detector_mode": "isolation_forest",
                }
            ]
        ).to_csv(results_dir / "results.csv", index=False)

        pd.DataFrame(
            [
                {
                    "config_name": "baseline",
                    "MAE": 1.1,
                    "MSE": 2.1,
                    "RMSE": 1.449,
                    "R2": 0.7,
                    "kalman_enabled": False,
                    "anomaly_enabled": False,
                    "llm_enabled": False,
                    "model_type": "residual_linear",
                }
            ]
        ).to_csv(results_dir / "ablation_results.csv", index=False)

        pd.DataFrame(
            [
                {
                    "config_name": "+llm",
                    "simplify_ui": 10,
                    "highlight_relevant": 20,
                    "no_change": 30,
                    "request_human_review": 0,
                    "total_decisions": 60,
                }
            ]
        ).to_csv(results_dir / "llm_decision_summary.csv", index=False)

    def test_tables_generated_when_files_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            user_study_dir = root / "user_study"
            output_dir = root / "paper_tables"
            results_dir.mkdir()
            user_study_dir.mkdir()
            self._write_core_inputs(results_dir)
            pd.DataFrame([{"condition": "adaptive", "metric": 1.2}]).to_csv(
                user_study_dir / "condition_comparison.csv", index=False
            )

            generated, skipped = generate_tables(
                results_dir=results_dir,
                user_study_dir=user_study_dir,
                output_dir=output_dir,
                strict=False,
            )

            self.assertEqual(len(generated), 4)
            self.assertEqual(skipped, [])
            self.assertTrue((output_dir / "table1_model_metrics.csv").exists())
            self.assertTrue((output_dir / "table2_ablation_study.csv").exists())
            self.assertTrue((output_dir / "table3_llm_decision_summary.csv").exists())
            self.assertTrue((output_dir / "table4_user_study_condition_comparison.csv").exists())

    def test_missing_user_study_comparison_skipped_non_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            user_study_dir = root / "user_study"
            output_dir = root / "paper_tables"
            results_dir.mkdir()
            user_study_dir.mkdir()
            self._write_core_inputs(results_dir)

            generated, skipped = generate_tables(
                results_dir=results_dir,
                user_study_dir=user_study_dir,
                output_dir=output_dir,
                strict=False,
            )

            self.assertEqual(len(generated), 3)
            self.assertTrue(any("table4_user_study_condition_comparison.csv skipped" in item for item in skipped))
            self.assertFalse((output_dir / "table4_user_study_condition_comparison.csv").exists())

    def test_missing_user_study_comparison_fails_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            user_study_dir = root / "user_study"
            output_dir = root / "paper_tables"
            results_dir.mkdir()
            user_study_dir.mkdir()
            self._write_core_inputs(results_dir)

            with self.assertRaises(ValueError):
                generate_tables(
                    results_dir=results_dir,
                    user_study_dir=user_study_dir,
                    output_dir=output_dir,
                    strict=True,
                )

    def test_no_fake_nasa_tlx_tables_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            user_study_dir = root / "user_study"
            output_dir = root / "paper_tables"
            results_dir.mkdir()
            user_study_dir.mkdir()
            self._write_core_inputs(results_dir)

            generate_tables(
                results_dir=results_dir,
                user_study_dir=user_study_dir,
                output_dir=output_dir,
                strict=False,
            )

            self.assertFalse((output_dir / "table2_nasa_tlx_components.csv").exists())
            self.assertFalse((output_dir / "table120_participant_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
