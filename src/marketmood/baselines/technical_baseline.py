"""Deterministic technical-analysis baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketmood.labels import LABEL_ORDER
from marketmood.metrics import classification_metrics
from marketmood.models.base import MarketMoodModel


def predict_momentum_class(ret_5d: float, vol_20d: float, threshold: float = 0.75) -> str:
    """Predict abnormal-move class from trailing momentum over volatility."""
    if pd.isna(ret_5d) or pd.isna(vol_20d) or vol_20d <= 0:
        return "neutral"
    score = ret_5d / vol_20d
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


@dataclass
class TechnicalAnalysisBaseline(MarketMoodModel):
    """Volatility-adjusted range-breakout rule using recent price action only."""

    threshold: float = 0.75
    upper_range: float = 0.8
    lower_range: float = 0.2
    name: str = "technical_analysis_baseline"

    @staticmethod
    def required_columns() -> list[str]:
        """Return columns required for inference."""
        return [
            "ret_5d",
            "vol_20d",
            "range_position_20d",
            "breakout_strength_20d",
            "breakdown_strength_20d",
        ]

    def momentum_score(self, frame: pd.DataFrame) -> pd.Series:
        """Compute trailing momentum normalized by trailing volatility."""
        self._validate_columns(frame)
        score = frame["ret_5d"] / frame["vol_20d"]
        return score.replace([float("inf"), float("-inf")], pd.NA)

    def breakout_score(self, frame: pd.DataFrame) -> pd.Series:
        """Compute prior close's breakout strength normalized by trailing volatility."""
        self._validate_columns(frame)
        score = frame["breakout_strength_20d"] / frame["vol_20d"]
        return score.replace([float("inf"), float("-inf")], pd.NA)

    def breakdown_score(self, frame: pd.DataFrame) -> pd.Series:
        """Compute prior close's breakdown strength normalized by trailing volatility."""
        self._validate_columns(frame)
        score = frame["breakdown_strength_20d"] / frame["vol_20d"]
        return score.replace([float("inf"), float("-inf")], pd.NA)

    def predict(self, frame: pd.DataFrame) -> list[str]:
        """Predict labels from a simple volatility-adjusted breakout setup."""
        self._validate_columns(frame)
        signals = pd.DataFrame(
            {
                "momentum_score": self.momentum_score(frame),
                "breakout_score": self.breakout_score(frame),
                "breakdown_score": self.breakdown_score(frame),
                "range_position_20d": frame["range_position_20d"],
            }
        )
        return [self._label_from_signal(row) for _, row in signals.iterrows()]

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return deterministic one-hot probabilities for the rule output."""
        predictions = self.predict(frame)
        probabilities = pd.DataFrame(0.0, index=frame.index, columns=LABEL_ORDER)
        for index, label in zip(frame.index, predictions, strict=True):
            probabilities.loc[index, label] = 1.0
        return probabilities

    def _label_from_signal(self, row: pd.Series) -> str:
        momentum_score = row["momentum_score"]
        breakout_score = row["breakout_score"]
        breakdown_score = row["breakdown_score"]
        range_position = row["range_position_20d"]
        if pd.isna(momentum_score) or pd.isna(range_position):
            return "neutral"

        positive_score = max(
            value for value in [momentum_score, breakout_score] if not pd.isna(value)
        )
        negative_score = min(
            value for value in [momentum_score, breakdown_score] if not pd.isna(value)
        )

        if range_position >= self.upper_range and positive_score > self.threshold:
            return "positive"
        if range_position <= self.lower_range and negative_score < -self.threshold:
            return "negative"
        return "neutral"

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = set(self.required_columns()).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required baseline columns: {sorted(missing)}")


def build_prediction_frame(frame: pd.DataFrame, model: TechnicalAnalysisBaseline) -> pd.DataFrame:
    """Create a report-ready prediction frame for one split."""
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame).add_prefix("prob_")
    output_columns = [
        "id",
        "split",
        "ticker",
        "post_date",
        "event_date",
        "original",
        "target",
        "ret_5d",
        "vol_20d",
        "range_position_20d",
        "breakout_strength_20d",
        "breakdown_strength_20d",
    ]
    available_columns = [column for column in output_columns if column in frame.columns]
    output = frame[available_columns].copy()
    output["momentum_score"] = model.momentum_score(frame).to_numpy()
    output["breakout_score"] = model.breakout_score(frame).to_numpy()
    output["breakdown_score"] = model.breakdown_score(frame).to_numpy()
    output["predicted_label"] = predictions
    output = pd.concat([output.reset_index(drop=True), probabilities.reset_index(drop=True)], axis=1)
    return output.rename(columns={"target": "true_label"})


def evaluate_technical_baseline(
    modeling_dataset_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    threshold: float,
    upper_range: float,
    lower_range: float,
    split: str = "test",
) -> dict[str, object]:
    """Evaluate the technical-analysis baseline and persist predictions/metrics."""
    frame = pd.read_csv(modeling_dataset_path)
    evaluation_frame = frame.loc[frame["split"].eq(split)].copy()
    if evaluation_frame.empty:
        raise ValueError(f"No rows found for split: {split}")

    model = TechnicalAnalysisBaseline(
        threshold=threshold,
        upper_range=upper_range,
        lower_range=lower_range,
    )
    prediction_frame = build_prediction_frame(evaluation_frame, model)
    metrics = classification_metrics(
        prediction_frame["true_label"].tolist(),
        prediction_frame["predicted_label"].tolist(),
        labels=LABEL_ORDER,
    )
    metrics["model_name"] = model.name
    metrics["split"] = split
    metrics["threshold"] = threshold
    metrics["upper_range"] = upper_range
    metrics["lower_range"] = lower_range
    metrics["n_rows"] = int(len(prediction_frame))

    predictions_output = Path(predictions_path)
    metrics_output = Path(metrics_path)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_output, index=False)
    with metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics
