"""Optional behavior anomaly detection backed by Isolation Forest."""

from typing import Any

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    IsolationForest = None
    SKLEARN_AVAILABLE = False


class BehaviorAnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42, fallback_mode: str = "error"):
        self.fallback_mode = fallback_mode
        self.model: Any = None
        self.is_fallback = False
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
            )
        elif fallback_mode == "zscore":
            self.is_fallback = True
            self._mean = None
            self._std = None
        else:
            raise ImportError(
                "scikit-learn is required for IsolationForest. "
                "Use fallback_mode='zscore' for explicit deterministic fallback."
            )

    def fit(self, x):
        x_arr = np.asarray(x, dtype=float)
        if self.is_fallback:
            self._mean = np.mean(x_arr, axis=0)
            self._std = np.std(x_arr, axis=0) + 1e-6
            return self
        self.model.fit(x_arr)
        return self

    def score(self, x):
        x_arr = np.asarray(x, dtype=float)
        if self.is_fallback:
            z = np.abs((x_arr - self._mean) / self._std)
            return np.max(z, axis=1)
        # Convert so higher = more anomalous
        return -self.model.score_samples(x_arr)

    def predict(self, x):
        x_arr = np.asarray(x, dtype=float)
        if self.is_fallback:
            scores = self.score(x_arr)
            return np.where(scores > 3.0, -1, 1)
        return self.model.predict(x_arr)
