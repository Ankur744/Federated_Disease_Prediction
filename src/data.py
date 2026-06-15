from __future__ import annotations

from dataclasses import dataclass
import numpy as np


FEATURE_NAMES = [
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "insulin",
    "family_history",
    "physical_activity",
]


@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


def generate_healthcare_dataset(samples: int = 2500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    age = rng.normal(48, 13, samples).clip(18, 85)
    bmi = rng.normal(27, 5, samples).clip(16, 45)
    blood_pressure = rng.normal(125, 16, samples).clip(80, 190)
    glucose = rng.normal(115, 35, samples).clip(60, 260)
    cholesterol = rng.normal(205, 42, samples).clip(110, 340)
    insulin = rng.normal(85, 38, samples).clip(10, 260)
    family_history = rng.binomial(1, 0.36, samples)
    physical_activity = rng.normal(4.2, 2.0, samples).clip(0, 10)

    x = np.column_stack(
        [
            age,
            bmi,
            blood_pressure,
            glucose,
            cholesterol,
            insulin,
            family_history,
            physical_activity,
        ]
    )

    risk_score = (
        0.035 * (age - 45)
        + 0.09 * (bmi - 25)
        + 0.018 * (blood_pressure - 120)
        + 0.026 * (glucose - 105)
        + 0.011 * (cholesterol - 190)
        + 0.012 * (insulin - 75)
        + 0.9 * family_history
        - 0.22 * physical_activity
        + rng.normal(0, 1.0, samples)
    )
    probabilities = 1.0 / (1.0 + np.exp(-risk_score / 3.0))
    y = rng.binomial(1, probabilities)

    return x.astype(float), y.astype(int)


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    test_size = int(len(y) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def fit_standard_scaler(x_train: np.ndarray) -> StandardScaler:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return StandardScaler(mean, std)


def split_clients(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    non_iid_strength: float = 0.75,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(y))
    sorted_order = np.argsort(y + rng.normal(0, 1 - non_iid_strength, len(y)))

    mixed_order = []
    for random_idx, sorted_idx in zip(random_order, sorted_order):
        mixed_order.append(sorted_idx if rng.random() < non_iid_strength else random_idx)

    chunks = np.array_split(np.array(mixed_order), n_clients)
    return [(x[chunk], y[chunk]) for chunk in chunks if len(chunk) > 0]

