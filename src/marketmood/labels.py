"""Target-label construction for abnormal stock moves."""

from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_ORDER = ["negative", "neutral", "positive"]


def label_from_abnormal_score(score: float, threshold: float) -> str:
    """Map an abnormal-return score to the three-class target."""
    if pd.isna(score) or not np.isfinite(score):
        raise ValueError("Cannot label a missing or non-finite abnormal score.")
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"
