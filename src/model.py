from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass
class LogisticRegressionModel:
    n_features: int
    learning_rate: float = 0.05
    l2: float = 0.001

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=float)
        self.bias = 0.0

    def clone(self) -> "LogisticRegressionModel":
        copied = LogisticRegressionModel(self.n_features, self.learning_rate, self.l2)
        copied.set_parameters(self.get_parameters())
        return copied

    def get_parameters(self) -> tuple[np.ndarray, float]:
        return self.weights.copy(), float(self.bias)

    def set_parameters(self, parameters: tuple[np.ndarray, float]) -> None:
        weights, bias = parameters
        self.weights = weights.copy()
        self.bias = float(bias)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 32) -> None:
        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx]

                predictions = self.predict_proba(xb)
                error = predictions - yb

                grad_w = (xb.T @ error) / len(yb) + self.l2 * self.weights
                grad_b = float(np.mean(error))

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

