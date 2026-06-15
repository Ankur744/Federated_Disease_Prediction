from __future__ import annotations

import argparse

from src.data import (
    FEATURE_NAMES,
    fit_standard_scaler,
    generate_healthcare_dataset,
    split_clients,
    train_test_split,
)
from src.federated import train_federated
from src.metrics import binary_classification_metrics, print_metrics
from src.model import LogisticRegressionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated disease prediction thesis prototype")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--dp-noise", type=float, default=0.01)
    parser.add_argument("--non-iid-strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_centralized(x_train, y_train, x_test, y_test, learning_rate: float) -> dict[str, float]:
    model = LogisticRegressionModel(n_features=x_train.shape[1], learning_rate=learning_rate)
    model.train(x_train, y_train, epochs=80)
    scores = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    return binary_classification_metrics(y_test, predictions, scores)


def main() -> None:
    args = parse_args()

    x, y = generate_healthcare_dataset(samples=args.samples, seed=args.seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=args.seed)

    scaler = fit_standard_scaler(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clients = split_clients(
        x_train,
        y_train,
        n_clients=args.clients,
        non_iid_strength=args.non_iid_strength,
        seed=args.seed,
    )

    print("Federated Learning-Based Privacy-Preserving Disease Prediction System")
    print("=" * 70)
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Training samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"Clients/hospitals: {len(clients)} | DP noise: {args.dp_noise}")
    print("\nClient data distribution:")
    for idx, (_, y_client) in enumerate(clients, start=1):
        disease_rate = y_client.mean()
        print(f"  Hospital {idx}: records={len(y_client):4d}, positive_rate={disease_rate:.3f}")

    print("\nTraining federated model...")
    federated_model, history = train_federated(
        clients=clients,
        x_test=x_test,
        y_test=y_test,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        dp_noise=args.dp_noise,
        seed=args.seed,
    )

    federated_scores = federated_model.predict_proba(x_test)
    federated_predictions = federated_model.predict(x_test)
    federated_metrics = binary_classification_metrics(y_test, federated_predictions, federated_scores)

    centralized_metrics = train_centralized(x_train, y_train, x_test, y_test, args.learning_rate)

    print_metrics("Federated Model Final Metrics", federated_metrics)
    print_metrics("Centralized Baseline Metrics", centralized_metrics)

    print("\nConclusion")
    print("----------")
    print(
        "The federated model trains across simulated hospitals without sharing raw patient data. "
        "Compare its metrics against the centralized baseline to discuss the privacy-accuracy trade-off."
    )


if __name__ == "__main__":
    main()

