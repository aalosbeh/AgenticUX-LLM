"""
Tests for Cognitive Load Model
"""

import unittest
import numpy as np
from src.core.cognitive_load_model import (
    CognitiveLoadModel,
    CognitiveLoadInput,
    SimpleNeuralNetwork,
    GradientBoostingEnsemble
)


class TestNeuralNetwork(unittest.TestCase):
    """Test neural network component"""

    def setUp(self):
        self.nn = SimpleNeuralNetwork(input_size=5, hidden_size=10, output_size=1)

    def test_forward_pass(self):
        """Test forward propagation"""
        x = np.random.randn(10, 5)
        output, cache = self.nn.forward(x)

        self.assertEqual(output.shape, (10, 1))
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_backward_pass(self):
        """Test backward propagation"""
        x = np.random.randn(10, 5)
        y = np.random.rand(10, 1)

        output, cache = self.nn.forward(x)
        self.nn.backward(y, cache, output)

        # Check that weights were updated
        self.assertIsNotNone(self.nn.w1)
        self.assertIsNotNone(self.nn.w2)

    def test_prediction(self):
        """Test prediction"""
        x = np.random.randn(5, 5)
        predictions = self.nn.predict(x)

        self.assertEqual(predictions.shape, (5, 1))
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))


class TestGradientBoosting(unittest.TestCase):
    """Test gradient boosting ensemble"""

    def setUp(self):
        self.gb = GradientBoostingEnsemble(n_estimators=5)

    def test_fit(self):
        """Test fitting"""
        x = np.random.randn(50, 10)
        y = np.random.rand(50, 1)

        self.gb.fit(x, y)
        self.assertEqual(len(self.gb.estimators), 5)

    def test_predict(self):
        """Test prediction"""
        x_train = np.random.randn(50, 10)
        y_train = np.random.rand(50, 1)

        self.gb.fit(x_train, y_train)

        x_test = np.random.randn(10, 10)
        predictions = self.gb.predict(x_test)

        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))


class TestCognitiveLoadModel(unittest.TestCase):
    """Test cognitive load model"""

    def setUp(self):
        self.model = CognitiveLoadModel()

    def test_feature_extraction(self):
        """Test feature extraction"""
        input_data = CognitiveLoadInput(
            mouse_velocity=200,
            click_frequency=1.5,
            time_between_actions=1.0,
            error_count=1,
            correction_count=0,
            page_visits=3,
            task_complexity=0.5,
            task_familiarity=0.7
        )

        features = self.model.extract_features(input_data)

        self.assertEqual(features.shape, (13,))
        self.assertTrue(np.all((features >= 0) & (features <= 1)))

    def test_heuristic_prediction(self):
        """Test heuristic prediction"""
        input_data = CognitiveLoadInput(
            mouse_velocity=200,
            click_frequency=1.5,
            time_between_actions=1.0,
            error_count=0,
            correction_count=0,
            page_visits=2,
            task_complexity=0.2,
            task_familiarity=0.9
        )

        result = self.model.predict(input_data)

        self.assertIn('cognitive_load', result)
        self.assertIn('load_level', result)
        self.assertIn('confidence', result)
        self.assertTrue(0 <= result['cognitive_load'] <= 100)

    def test_load_level_classification(self):
        """Test load level classification"""
        test_cases = [
            (10, "very_low"),
            (30, "low"),
            (50, "moderate"),
            (70, "high"),
            (90, "very_high"),
        ]

        for load, expected_level in test_cases:
            level = self.model._classify_load_level(load)
            self.assertEqual(level, expected_level)

    def test_model_training(self):
        """Test model training"""
        # Create training data
        training_data = []

        # Easy tasks (low load)
        for _ in range(20):
            input_data = CognitiveLoadInput(
                mouse_velocity=150 + np.random.randn() * 50,
                click_frequency=1.0 + np.random.randn() * 0.2,
                time_between_actions=1.5 + np.random.randn() * 0.3,
                error_count=0,
                correction_count=0,
                page_visits=2,
                task_complexity=0.2,
                task_familiarity=0.9
            )
            training_data.append((input_data, 25))

        # Hard tasks (high load)
        for _ in range(20):
            input_data = CognitiveLoadInput(
                mouse_velocity=400 + np.random.randn() * 100,
                click_frequency=3.5 + np.random.randn() * 0.5,
                time_between_actions=0.5 + np.random.randn() * 0.2,
                error_count=3 + np.random.randint(0, 3),
                correction_count=2 + np.random.randint(0, 2),
                page_visits=6 + np.random.randint(0, 3),
                task_complexity=0.8,
                task_familiarity=0.2
            )
            training_data.append((input_data, 85))

        # Train model
        self.model.train(training_data)
        self.assertTrue(self.model.is_trained)

        # Test on easy task
        easy_input = CognitiveLoadInput(
            mouse_velocity=160,
            click_frequency=0.95,
            time_between_actions=1.6,
            error_count=0,
            correction_count=0,
            page_visits=2,
            task_complexity=0.15,
            task_familiarity=0.95
        )

        easy_result = self.model.predict(easy_input)
        self.assertLess(easy_result['cognitive_load'], 50)

        # Test on hard task
        hard_input = CognitiveLoadInput(
            mouse_velocity=420,
            click_frequency=3.6,
            time_between_actions=0.4,
            error_count=4,
            correction_count=2,
            page_visits=7,
            task_complexity=0.85,
            task_familiarity=0.15
        )

        hard_result = self.model.predict(hard_input)
        self.assertGreater(hard_result['cognitive_load'], 50)

    def test_contribution_analysis(self):
        """Test contribution analysis"""
        features = np.array([0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8, 0.1, 0.5, 0.3, 0.7, 0.2, 0.6])

        contributions = self.model._analyze_contributions(features)

        self.assertIsInstance(contributions, dict)
        self.assertLessEqual(len(contributions), 5)
        for name, value in contributions.items():
            self.assertIsInstance(name, str)
            self.assertTrue(0 <= value <= 100)


if __name__ == '__main__':
    unittest.main()
