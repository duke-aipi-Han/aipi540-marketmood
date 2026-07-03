"""Shared model interface for MarketMood implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd

from marketmood.labels import LABEL_ORDER


class MarketMoodModel(ABC):
    """Minimal interface shared by all MarketMood predictors."""

    name: str
    labels: tuple[str, ...] = tuple(LABEL_ORDER)

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit the model when training is needed."""
        return self

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> list[str]:
        """Predict one abnormal-move label per row."""

    @abstractmethod
    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return class-probability columns ordered by ``self.labels``."""
