"""Simple 1D Kalman filter for smoothing scalar observations."""


class KalmanFilter:
    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 2.0):
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.reset()

    def reset(self) -> None:
        self.x_estimate = 0.0
        self.p_estimate = 1.0
        self.initialized = False

    def update(self, measurement: float) -> float:
        z = float(measurement)
        if not self.initialized:
            self.x_estimate = z
            self.initialized = True
            return self.x_estimate

        p_pred = self.p_estimate + self.process_variance
        k_gain = p_pred / (p_pred + self.measurement_variance)
        self.x_estimate = self.x_estimate + k_gain * (z - self.x_estimate)
        self.p_estimate = (1.0 - k_gain) * p_pred
        return self.x_estimate
