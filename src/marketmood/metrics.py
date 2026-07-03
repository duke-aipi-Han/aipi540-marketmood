"""Shared classification metrics utilities."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from marketmood.labels import LABEL_ORDER


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, object]:
    """Return common metrics for model comparison."""
    metric_labels = labels or LABEL_ORDER
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=metric_labels, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=metric_labels, average="weighted")),
        "labels": metric_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=metric_labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=metric_labels,
            output_dict=True,
            zero_division=0,
        ),
    }
