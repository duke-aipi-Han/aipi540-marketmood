"""Deterministic technical-analysis baseline."""

from __future__ import annotations


def predict_momentum_class(ret_5d: float, vol_20d: float, threshold: float = 0.5) -> str:
    """Predict abnormal-move class from trailing momentum over volatility."""
    if vol_20d <= 0:
        return "neutral"
    score = ret_5d / vol_20d
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"
