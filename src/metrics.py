from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_score),
    }
    result.update({key: float(value) for key, value in cm.items()})
    return result


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return 0.5

    wins = 0.0
    total = len(positive_scores) * len(negative_scores)
    for pos_score in positive_scores:
        wins += np.sum(pos_score > negative_scores)
        wins += 0.5 * np.sum(pos_score == negative_scores)
    return float(wins / total)


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:>10}: {metrics[key]:.4f}")
    print(f"confusion: TP={int(metrics['tp'])}, TN={int(metrics['tn'])}, FP={int(metrics['fp'])}, FN={int(metrics['fn'])}")

