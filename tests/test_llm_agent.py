"""Tests for optional LLM agent module."""

import unittest

from src.agents.llm_agent import LLMAgent


class TestLLMAgent(unittest.TestCase):
    def test_mock_mode_output_schema(self):
        agent = LLMAgent()
        result = agent.analyze_user_state(
            {
                "error_count": 4,
                "task_complexity": 0.85,
                "time_pressure": 0.9,
            }
        )
        self.assertIn(result["action"], {"simplify_ui", "highlight_relevant", "no_change", "request_human_review"})
        self.assertIsInstance(result["reasoning"], str)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertIn(result["mode"], {"openai", "mock"})

    def test_mock_mode_is_deterministic(self):
        agent = LLMAgent()
        features = {
            "error_count": 2,
            "task_complexity": 0.75,
            "time_pressure": 0.2,
            "anomaly_score": 0.95,
            "predicted_cognitive_load": 78.0,
        }
        first = agent.analyze_user_state(features)
        second = agent.analyze_user_state(features)
        self.assertEqual(first, second)

    def test_schema_keys_always_present(self):
        agent = LLMAgent()
        result = agent.analyze_user_state({})
        self.assertIn("action", result)
        self.assertIn("reasoning", result)
        self.assertIn("confidence", result)
        self.assertIn("mode", result)
        self.assertIn(result["action"], LLMAgent.VALID_ACTIONS)
        self.assertTrue(0.0 <= float(result["confidence"]) <= 1.0)


if __name__ == "__main__":
    unittest.main()
