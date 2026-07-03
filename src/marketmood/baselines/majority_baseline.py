"""Majority-class baseline helpers."""

from __future__ import annotations

from collections import Counter


def majority_label(labels: list[str]) -> str:
    """Return the most common label from a training split."""
    if not labels:
        raise ValueError("Cannot compute majority label from an empty label list.")
    return Counter(labels).most_common(1)[0][0]
