"""Shared classification metrics utilities."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    """Return common metrics for model comparison."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }
