"""
Cognitive Load Model for Agentic UX System
ML model for real-time cognitive load assessment using gradient boosting and neural networks.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CognitiveLoadInput:
    """Input features for cognitive load prediction"""
    # Behavioral features
    mouse_velocity: float  # pixels/second
    click_frequency: float  # clicks/second
    time_between_actions: float  # seconds
    error_count: int
    correction_count: int
    page_visits: int

    # Physiological features (if available)
    heart_rate: Optional[float] = None  # BPM
    pupil_dilation: Optional[float] = None  # mm
    blink_rate: Optional[float] = None  # blinks/minute

    # Task context
    task_complexity: float = 0.5  # 0-1
    task_familiarity: float = 0.5  # 0-1
    time_pressure: float = 0.5  # 0-1

    # UI context
    element_density: float = 0.5  # 0-1
    color_complexity: float = 0.5  # 0-1


class SimpleNeuralNetwork:
    """Simple feedforward neural network for cognitive load prediction"""

    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = 0.01

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass"""
        # Hidden layer
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.relu(z1)

        # Output layer
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)

        cache = {"z1": z1, "a1": a1, "z2": z2, "x": x}
        return a2, cache

    def backward(self, y: np.ndarray, cache: Dict[str, np.ndarray], output: np.ndarray) -> None:
        """Backward pass"""
        m = y.shape[0]

        # Output layer gradient
        dz2 = output - y
        dw2 = np.dot(cache["a1"].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradient
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (cache["a1"] > 0)
        dw1 = np.dot(cache["x"].T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make prediction"""
        output, _ = self.forward(x)
        return output


class GradientBoostingEnsemble:
    """Simple gradient boosting ensemble for cognitive load prediction"""

    def __init__(self, n_estimators: int = 5):
        self.n_estimators = n_estimators
        self.estimators = []
        self.learning_rate = 0.1
        self.initial_prediction = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit ensemble"""
        # Initialize residuals
        residuals = y - np.full_like(y, self.initial_prediction)

        for i in range(self.n_estimators):
            # Create simple linear estimator
            weights = np.linalg.lstsq(x, residuals, rcond=None)[0]
            self.estimators.append(weights)

            # Update residuals
            predictions = np.dot(x, weights)
            residuals -= self.learning_rate * predictions

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make prediction"""
        predictions = np.full(x.shape[0], self.initial_prediction)

        for weights in self.estimators:
            predictions += self.learning_rate * np.dot(x, weights)

        # Clip to [0, 1] range
        return np.clip(predictions, 0, 1)


class CognitiveLoadModel:
    """
    Ensemble model combining gradient boosting and neural networks
    for real-time cognitive load assessment.
    """

    def __init__(self):
        self.nn_model = SimpleNeuralNetwork(input_size=13)  # 13 features
        self.gb_model = GradientBoostingEnsemble(n_estimators=5)
        self.is_trained = False
        self.feature_scaler = None
        self.training_data = []
        self.training_labels = []

        # Thresholds for load levels
        self.thresholds = {
            "low": 0.3,
            "moderate": 0.6,
            "high": 0.8,
            "very_high": 0.95
        }

    def extract_features(self, input_data: CognitiveLoadInput) -> np.ndarray:
        """Extract feature vector from input"""
        features = np.array([
            input_data.mouse_velocity / 1000,  # Normalize
            input_data.click_frequency / 5,
            input_data.time_between_actions / 5,
            input_data.error_count / 10,
            input_data.correction_count / 10,
            input_data.page_visits / 10,
            input_data.heart_rate / 200 if input_data.heart_rate else 0.5,
            input_data.pupil_dilation / 8 if input_data.pupil_dilation else 0.5,
            input_data.blink_rate / 30 if input_data.blink_rate else 0.5,
            input_data.task_complexity,
            input_data.task_familiarity,
            input_data.time_pressure,
            (input_data.element_density + input_data.color_complexity) / 2
        ])

        return np.clip(features, 0, 1)

    def train(self, training_data: List[Tuple[CognitiveLoadInput, float]]) -> None:
        """Train both models on labeled data"""
        logger.info(f"Training cognitive load model on {len(training_data)} samples")

        x_list = []
        y_list = []

        for input_data, label in training_data:
            features = self.extract_features(input_data)
            x_list.append(features)
            y_list.append(label / 100)  # Normalize to 0-1

        x = np.array(x_list)
        y = np.array(y_list).reshape(-1, 1)

        # Train gradient boosting
        self.gb_model.fit(x, y)

        # Train neural network
        for epoch in range(100):
            output, cache = self.nn_model.forward(x)
            self.nn_model.backward(y, cache, output)

        self.is_trained = True
        logger.info("Model training complete")

    def predict(self, input_data: CognitiveLoadInput) -> Dict[str, Any]:
        """
        Predict cognitive load using ensemble of models.
        Returns both predictions and confidence scores.
        """
        if not self.is_trained:
            # Use heuristic prediction if model not trained
            return self._heuristic_prediction(input_data)

        features = self.extract_features(input_data)
        features_batch = features.reshape(1, -1)

        # Get predictions from both models
        gb_pred = self.gb_model.predict(features_batch)[0]
        nn_pred, _ = self.nn_model.forward(features_batch)
        nn_pred = nn_pred[0, 0]

        # Ensemble prediction (weighted average)
        ensemble_pred = 0.6 * gb_pred + 0.4 * nn_pred
        normalized_load = ensemble_pred * 100

        # Determine load level
        load_level = self._classify_load_level(normalized_load)

        # Calculate confidence
        confidence = self._calculate_confidence(features)

        return {
            "cognitive_load": normalized_load,
            "load_level": load_level,
            "gb_prediction": gb_pred * 100,
            "nn_prediction": nn_pred * 100,
            "confidence": confidence,
            "component_contributions": self._analyze_contributions(features)
        }

    def _heuristic_prediction(self, input_data: CognitiveLoadInput) -> Dict[str, Any]:
        """
        Heuristic prediction when model not trained.
        Uses domain knowledge to estimate cognitive load.
        """
        # Behavioral component
        behavioral_load = min(
            (input_data.click_frequency / 5) * 30 +
            (input_data.error_count / 5) * 20 +
            (1 - input_data.time_between_actions / 5) * 20,
            100
        )

        # Task complexity component
        task_load = (
            input_data.task_complexity * 30 +
            (1 - input_data.task_familiarity) * 25 +
            input_data.time_pressure * 15
        )

        # UI component
        ui_load = (
            input_data.element_density * 20 +
            input_data.color_complexity * 10
        )

        overall_load = (behavioral_load + task_load + ui_load) / 3

        load_level = self._classify_load_level(overall_load)

        return {
            "cognitive_load": overall_load,
            "load_level": load_level,
            "gb_prediction": behavioral_load,
            "nn_prediction": task_load,
            "confidence": 0.6,
            "component_contributions": {
                "behavioral": behavioral_load,
                "task": task_load,
                "ui": ui_load
            }
        }

    def _classify_load_level(self, load_score: float) -> str:
        """Classify load into categorical level"""
        if load_score < self.thresholds["low"]:
            return "very_low"
        elif load_score < self.thresholds["moderate"]:
            return "low"
        elif load_score < self.thresholds["high"]:
            return "moderate"
        elif load_score < self.thresholds["very_high"]:
            return "high"
        else:
            return "very_high"

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in prediction"""
        # Higher confidence when features are in normal ranges
        feature_variance = np.var(features)
        confidence = max(0.5, min(1.0, 1.0 - feature_variance))
        return confidence

    def _analyze_contributions(self, features: np.ndarray) -> Dict[str, float]:
        """Analyze which features contribute most to load"""
        feature_names = [
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
            "ui_complexity"
        ]

        # Feature importance approximation
        contributions = {}
        for i, name in enumerate(feature_names):
            contributions[name] = float(features[i]) * 100

        # Sort by contribution
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return {name: score for name, score in sorted_contrib[:5]}

    def update_with_feedback(self, input_data: CognitiveLoadInput, actual_load: float) -> None:
        """Update model with user feedback"""
        self.training_data.append(input_data)
        self.training_labels.append(actual_load)

        # Retrain periodically
        if len(self.training_data) % 10 == 0:
            training_pairs = list(zip(self.training_data, self.training_labels))
            self.train(training_pairs)

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            "is_trained": self.is_trained,
            "training_samples": len(self.training_data),
            "model_type": "ensemble",
            "components": ["gradient_boosting", "neural_network"],
            "ensemble_weights": {"gb": 0.6, "nn": 0.4}
        }


# Example usage and testing
if __name__ == "__main__":
    model = CognitiveLoadModel()

    # Create sample training data
    training_data = []

    # Easy task: low cognitive load
    for i in range(20):
        input_data = CognitiveLoadInput(
            mouse_velocity=200 + np.random.randn() * 50,
            click_frequency=1.0 + np.random.randn() * 0.2,
            time_between_actions=1.5 + np.random.randn() * 0.3,
            error_count=0,
            correction_count=0,
            page_visits=2,
            heart_rate=70 + np.random.randn() * 5,
            pupil_dilation=3.0 + np.random.randn() * 0.5,
            task_complexity=0.2,
            task_familiarity=0.9,
            time_pressure=0.1
        )
        training_data.append((input_data, 25))

    # Hard task: high cognitive load
    for i in range(20):
        input_data = CognitiveLoadInput(
            mouse_velocity=400 + np.random.randn() * 100,
            click_frequency=3.5 + np.random.randn() * 0.5,
            time_between_actions=0.5 + np.random.randn() * 0.2,
            error_count=3 + np.random.randint(0, 3),
            correction_count=2 + np.random.randint(0, 2),
            page_visits=6 + np.random.randint(0, 3),
            heart_rate=95 + np.random.randn() * 10,
            pupil_dilation=5.5 + np.random.randn() * 0.8,
            task_complexity=0.8,
            task_familiarity=0.2,
            time_pressure=0.8
        )
        training_data.append((input_data, 85))

    # Train model
    model.train(training_data)

    # Test predictions
    print("Testing cognitive load model:")
    print("-" * 60)

    # Test low load scenario
    low_load_input = CognitiveLoadInput(
        mouse_velocity=180,
        click_frequency=0.9,
        time_between_actions=1.8,
        error_count=0,
        correction_count=0,
        page_visits=2,
        task_complexity=0.1,
        task_familiarity=0.95
    )

    result = model.predict(low_load_input)
    print(f"Low load scenario:")
    print(f"  Predicted load: {result['cognitive_load']:.1f}")
    print(f"  Level: {result['load_level']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Test high load scenario
    high_load_input = CognitiveLoadInput(
        mouse_velocity=450,
        click_frequency=4.2,
        time_between_actions=0.4,
        error_count=5,
        correction_count=3,
        page_visits=8,
        task_complexity=0.9,
        task_familiarity=0.1,
        time_pressure=0.9
    )

    result = model.predict(high_load_input)
    print(f"\nHigh load scenario:")
    print(f"  Predicted load: {result['cognitive_load']:.1f}")
    print(f"  Level: {result['load_level']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    print(f"\nModel stats: {model.get_model_stats()}")
