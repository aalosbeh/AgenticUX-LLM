import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_STUDY_SCRIPT = PROJECT_ROOT / "experiments" / "user_study_template.py"


class TestUserStudyTemplate(unittest.TestCase):
    def test_demo_template_creates_headers_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(USER_STUDY_SCRIPT),
                "--demo-template",
                "--output-dir",
                tmpdir,
            ]
            subprocess.run(cmd, check=True)

            template_path = Path(tmpdir) / "user_study_template.csv"
            self.assertTrue(template_path.exists())

            loaded = pd.read_csv(template_path)
            self.assertEqual(len(loaded), 0)
            self.assertEqual(
                list(loaded.columns),
                [
                    "participant_id",
                    "condition",
                    "task_id",
                    "completion_time_seconds",
                    "error_count",
                    "satisfaction_score_1_to_5",
                    "notes",
                ],
            )

    def test_invalid_schema_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad_schema.csv"
            pd.DataFrame(
                [
                    {
                        "participant_id": "P1",
                        "condition": "baseline",
                        "task_id": "T1",
                        "completion_time_seconds": 10.0,
                        "error_count": 0,
                    }
                ]
            ).to_csv(csv_path, index=False)

            cmd = [
                sys.executable,
                str(USER_STUDY_SCRIPT),
                "--input-csv",
                str(csv_path),
                "--output-dir",
                tmpdir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Input CSV is missing required columns: satisfaction_score_1_to_5",
                result.stderr,
            )

    def test_invalid_values_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad_values.csv"
            pd.DataFrame(
                [
                    {
                        "participant_id": "P1",
                        "condition": "wrong",
                        "task_id": "T1",
                        "completion_time_seconds": -1,
                        "error_count": -2,
                        "satisfaction_score_1_to_5": 6,
                        "notes": "",
                    }
                ]
            ).to_csv(csv_path, index=False)

            cmd = [
                sys.executable,
                str(USER_STUDY_SCRIPT),
                "--input-csv",
                str(csv_path),
                "--output-dir",
                tmpdir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid condition value(s): wrong", result.stderr)

    def test_valid_tiny_csv_produces_summary_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "valid.csv"
            pd.DataFrame(
                [
                    {
                        "participant_id": "P1",
                        "condition": "baseline",
                        "task_id": "T1",
                        "completion_time_seconds": 12.0,
                        "error_count": 1,
                        "satisfaction_score_1_to_5": 4,
                        "notes": "ok",
                    },
                    {
                        "participant_id": "P2",
                        "condition": "adaptive",
                        "task_id": "T1",
                        "completion_time_seconds": 10.0,
                        "error_count": 0,
                        "satisfaction_score_1_to_5": 5,
                        "notes": "",
                    },
                ]
            ).to_csv(csv_path, index=False)

            cmd = [
                sys.executable,
                str(USER_STUDY_SCRIPT),
                "--input-csv",
                str(csv_path),
                "--output-dir",
                tmpdir,
            ]
            subprocess.run(cmd, check=True)

            summary_path = Path(tmpdir) / "user_study_summary.csv"
            comparison_path = Path(tmpdir) / "condition_comparison.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(comparison_path.exists())

            summary_df = pd.read_csv(summary_path)
            comparison_df = pd.read_csv(comparison_path)
            self.assertEqual(int(summary_df.loc[0, "total_rows"]), 2)
            self.assertEqual(set(comparison_df["condition"]), {"baseline", "adaptive"})


if __name__ == "__main__":
    unittest.main()
