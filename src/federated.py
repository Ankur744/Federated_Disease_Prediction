from __future__ import annotations

import numpy as np

from src.metrics import binary_classification_metrics
from src.model import LogisticRegressionModel


def federated_average(
    parameters: list[tuple[np.ndarray, float]],
    client_sizes: list[int],
) -> tuple[np.ndarray, float]:
    total = sum(client_sizes)
    avg_weights = sum(weights * (size / total) for (weights, _), size in zip(parameters, client_sizes))
    avg_bias = sum(bias * (size / total) for (_, bias), size in zip(parameters, client_sizes))
    return avg_weights, float(avg_bias)


def add_differential_privacy_noise(
    parameters: tuple[np.ndarray, float],
    noise_multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if noise_multiplier <= 0:
        return parameters

    weights, bias = parameters
    noisy_weights = weights + rng.normal(0, noise_multiplier, size=weights.shape)
    noisy_bias = bias + float(rng.normal(0, noise_multiplier))
    return noisy_weights, noisy_bias


def train_federated(
    clients: list[tuple[np.ndarray, np.ndarray]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 25,
    local_epochs: int = 3,
    learning_rate: float = 0.05,
    dp_noise: float = 0.0,
    seed: int = 42,
) -> tuple[LogisticRegressionModel, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    n_features = clients[0][0].shape[1]
    global_model = LogisticRegressionModel(n_features=n_features, learning_rate=learning_rate)
    history = []

    for round_no in range(1, rounds + 1):
        local_parameters = []
        client_sizes = []

        for x_client, y_client in clients:
            local_model = global_model.clone()
            local_model.train(x_client, y_client, epochs=local_epochs)
            params = add_differential_privacy_noise(local_model.get_parameters(), dp_noise, rng)
            local_parameters.append(params)
            client_sizes.append(len(y_client))

        global_model.set_parameters(federated_average(local_parameters, client_sizes))

        scores = global_model.predict_proba(x_test)
        predictions = (scores >= 0.5).astype(int)
        metrics = binary_classification_metrics(y_test, predictions, scores)
        metrics["round"] = float(round_no)
        history.append(metrics)

        if round_no == 1 or round_no % 5 == 0 or round_no == rounds:
            print(
                f"Round {round_no:02d} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"auc={metrics['roc_auc']:.4f}"
            )

    return global_model, history

