"""Inference helpers for trained MarketMood models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from marketmood.config import load_config
from marketmood.features import PRICE_FEATURE_COLUMNS, compute_price_feature_frame
from marketmood.labels import LABEL_ORDER
from marketmood.models.classical import ClassicalFeatureMode, ClassicalLogisticModel
from marketmood.models.deep_fusion import DeepModelConfig, PriceFeatureScaler, TransformerFusionClassifier
from marketmood.prices import load_cached_prices
from marketmood.text_processing import build_text_input
from marketmood.training.train_deep_fusion import (
    build_deep_prediction_frame,
    load_deep_artifact,
    select_torch_device,
)


@dataclass(frozen=True)
class SignalPrediction:
    """Prediction and historical context for one app request."""

    ticker: str
    event_date: pd.Timestamp
    feature_cutoff_date: pd.Timestamp
    target_end_date: pd.Timestamp | None
    message: str
    predicted_label: str
    probabilities: dict[str, float]
    price_features: dict[str, float]
    close_t: float | None
    close_t_plus_1: float | None
    known_target: str | None
    known_future_return_1d: float | None
    known_abnormal_score: float | None


@dataclass(frozen=True)
class LoadedDeepModel:
    """Loaded deep model components for single-row inference."""

    model: TransformerFusionClassifier
    tokenizer: Any
    price_scaler: PriceFeatureScaler | None
    config: DeepModelConfig
    device: torch.device


def _resolve_project_path(config_path: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


class MarketMoodInferenceService:
    """Build leakage-safe prediction rows from cached prices and user text."""

    def __init__(
        self,
        model: ClassicalLogisticModel,
        modeling_dataset: pd.DataFrame,
        price_cache_dir: str | Path,
        abnormal_threshold: float,
        text_format: str = "raw",
        classical_models: dict[str, ClassicalLogisticModel] | None = None,
        deep_models: dict[str, LoadedDeepModel] | None = None,
        official_model_name: str = "deep_text_price",
    ) -> None:
        self.model = model
        self.classical_models = classical_models or {"classical_text_price": model}
        self.deep_models = deep_models or {}
        self.official_model_name = official_model_name
        self.modeling_dataset = modeling_dataset.copy()
        self.modeling_dataset["event_date"] = pd.to_datetime(
            self.modeling_dataset["event_date"], errors="coerce"
        ).dt.normalize()
        self.price_cache_dir = Path(price_cache_dir)
        self.abnormal_threshold = abnormal_threshold
        self.text_format = text_format

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config.yaml",
        feature_mode: ClassicalFeatureMode = "text_price",
    ) -> "MarketMoodInferenceService":
        """Create the app inference service from project defaults."""
        config = load_config(config_path)
        model_dir = _resolve_project_path(config.source_path, config.values["paths"]["classical_model_dir"])
        model_path = model_dir / f"{feature_mode}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")

        classical_models = {
            f"classical_{artifact_path.stem}": ClassicalLogisticModel.load(artifact_path)
            for artifact_path in sorted(model_dir.glob("*.joblib"))
        }

        deep_models: dict[str, LoadedDeepModel] = {}
        deep_model_dir = _resolve_project_path(config.source_path, config.values["paths"]["deep_fusion_model_dir"])
        device = select_torch_device(str(config.values["project"].get("device_preference", "mps")))
        if deep_model_dir.exists():
            for artifact_dir in sorted(deep_model_dir.iterdir()):
                if not artifact_dir.is_dir() or not (artifact_dir / "model.pt").exists():
                    continue
                model, tokenizer, price_scaler, model_config = load_deep_artifact(artifact_dir, device)
                deep_models[f"deep_{model_config.feature_mode}"] = LoadedDeepModel(
                    model=model,
                    tokenizer=tokenizer,
                    price_scaler=price_scaler,
                    config=model_config,
                    device=device,
                )

        dataset_path = _resolve_project_path(config.source_path, config.values["paths"]["modeling_dataset"])
        price_cache_dir = _resolve_project_path(config.source_path, config.values["paths"]["price_cache_dir"])
        modeling_dataset = pd.read_csv(dataset_path)
        official_model_name = "deep_text_price" if "deep_text_price" in deep_models else f"classical_{feature_mode}"

        return cls(
            model=ClassicalLogisticModel.load(model_path),
            modeling_dataset=modeling_dataset,
            price_cache_dir=price_cache_dir,
            abnormal_threshold=float(config.values["labels"]["abnormal_threshold"]),
            text_format=str(config.values.get("features", {}).get("text_format", "raw")),
            classical_models=classical_models,
            deep_models=deep_models,
            official_model_name=official_model_name,
        )

    def available_tickers(self) -> list[str]:
        """Return tickers with modeled rows and cached prices."""
        tickers = sorted(self.modeling_dataset["ticker"].dropna().astype(str).unique())
        return [
            ticker
            for ticker in tickers
            if (self.price_cache_dir / f"{ticker.upper().replace('/', '-')}.csv").exists()
        ]

    def available_event_dates(self, ticker: str) -> list[str]:
        """Return available modeled event dates for one ticker."""
        rows = self._ticker_rows(ticker)
        dates = rows["event_date"].dropna().sort_values().dt.strftime("%Y-%m-%d").unique()
        return dates.tolist()

    def sample_posts(self, ticker: str, event_date: str | pd.Timestamp, limit: int = 25) -> list[str]:
        """Return historical posts for the selected ticker/date pair."""
        event_timestamp = pd.Timestamp(event_date).normalize()
        rows = self._ticker_rows(ticker)
        date_rows = rows.loc[rows["event_date"].eq(event_timestamp)].copy()
        if date_rows.empty:
            date_rows = rows.sort_values("event_date", ascending=False).head(limit)
        return date_rows["original"].dropna().astype(str).head(limit).tolist()

    def default_message(self, ticker: str, event_date: str | pd.Timestamp) -> str:
        """Return a representative message for a ticker/date pair."""
        posts = self.sample_posts(ticker, event_date, limit=1)
        if posts:
            return posts[0]
        return f"${ticker.upper()} "

    def predict(self, ticker: str, event_date: str | pd.Timestamp, message: str) -> SignalPrediction:
        """Predict with the official model for one ticker/date/message request."""
        predictions = self.predict_all(ticker, event_date, message)
        return predictions[self.official_model_name]

    def predict_all(
        self,
        ticker: str,
        event_date: str | pd.Timestamp,
        message: str,
    ) -> dict[str, SignalPrediction]:
        """Predict with all loaded app models for one ticker/date/message request."""
        event_timestamp = pd.Timestamp(event_date).normalize()
        feature_row = self._price_feature_row(ticker, event_timestamp)
        input_frame = self._build_input_frame(ticker, event_timestamp, message, feature_row)
        predictions: dict[str, SignalPrediction] = {}

        for model_name, model in self.classical_models.items():
            predicted_label = model.predict(input_frame)[0]
            probabilities = model.predict_proba(input_frame).iloc[0].to_dict()
            predictions[model_name] = self._signal_prediction_from_outputs(
                ticker=ticker,
                event_timestamp=event_timestamp,
                message=message,
                feature_row=feature_row,
                predicted_label=predicted_label,
                probabilities=probabilities,
            )

        for model_name, loaded_model in self.deep_models.items():
            prediction_frame = build_deep_prediction_frame(
                input_frame,
                model=loaded_model.model,
                tokenizer=loaded_model.tokenizer,
                model_config=loaded_model.config,
                price_scaler=loaded_model.price_scaler,
                device=loaded_model.device,
                batch_size=1,
            )
            prediction_row = prediction_frame.iloc[0]
            probabilities = {
                label: float(prediction_row[f"prob_{label}"])
                for label in LABEL_ORDER
            }
            predictions[model_name] = self._signal_prediction_from_outputs(
                ticker=ticker,
                event_timestamp=event_timestamp,
                message=message,
                feature_row=feature_row,
                predicted_label=str(prediction_row["predicted_label"]),
                probabilities=probabilities,
            )

        return predictions

    def _signal_prediction_from_outputs(
        self,
        ticker: str,
        event_timestamp: pd.Timestamp,
        message: str,
        feature_row: pd.Series,
        predicted_label: str,
        probabilities: dict[str, float],
    ) -> SignalPrediction:
        """Build a common prediction object from model outputs."""
        feature_cutoff_date = pd.Timestamp(feature_row["feature_cutoff_date"]).normalize()

        return SignalPrediction(
            ticker=ticker.upper(),
            event_date=event_timestamp,
            feature_cutoff_date=feature_cutoff_date,
            target_end_date=self._optional_timestamp(feature_row.get("target_end_date")),
            message=message,
            predicted_label=predicted_label,
            probabilities={label: float(probabilities[label]) for label in LABEL_ORDER},
            price_features={column: float(feature_row[column]) for column in PRICE_FEATURE_COLUMNS},
            close_t=self._optional_float(feature_row.get("close_t")),
            close_t_plus_1=self._optional_float(feature_row.get("close_t_plus_1")),
            known_target=self._optional_string(feature_row.get("target")),
            known_future_return_1d=self._optional_float(feature_row.get("future_return_1d")),
            known_abnormal_score=self._optional_float(feature_row.get("abnormal_score")),
        )

    def plot_price_context(
        self,
        prediction: SignalPrediction,
        lookback_rows: int = 45,
    ) -> Figure:
        """Render prior OHLC price context ending at the feature cutoff date."""
        ticker = prediction.ticker
        event_timestamp = prediction.event_date
        cutoff_date = prediction.feature_cutoff_date
        target_end_date = prediction.target_end_date or cutoff_date
        all_prices = self._price_history(ticker)
        prior_prices = all_prices.loc[all_prices["date"].le(cutoff_date)].tail(lookback_rows).copy()
        if prior_prices.empty:
            prices = prior_prices
        else:
            start_date = prior_prices["date"].min()
            prices = all_prices.loc[
                all_prices["date"].between(start_date, target_end_date)
            ].copy()

        figure, axis = plt.subplots(figsize=(9, 4.8))
        if prices.empty:
            axis.set_title(f"{ticker.upper()} price context unavailable")
            return figure

        dates = mdates.date2num(prices["date"])
        candle_width = 0.6
        for date_number, row in zip(dates, prices.itertuples(index=False), strict=True):
            color = "#2f7d5c" if row.close >= row.open else "#b24545"
            axis.vlines(date_number, row.low, row.high, color=color, linewidth=1.1, alpha=0.9)
            lower = min(row.open, row.close)
            height = abs(row.close - row.open)
            if height == 0:
                height = max(row.close * 0.001, 0.01)
            axis.add_patch(
                Rectangle(
                    (date_number - candle_width / 2, lower),
                    candle_width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.72,
                )
            )

        axis.axvline(
            mdates.date2num(event_timestamp),
            color="#334155",
            linestyle="--",
            linewidth=1.2,
            label="event date",
        )
        axis.axvspan(
            mdates.date2num(event_timestamp),
            mdates.date2num(target_end_date),
            color="#e2e8f0",
            alpha=0.35,
            label="historical outcome window",
        )
        label_colors = {
            "negative": "#b24545",
            "neutral": "#64748b",
            "positive": "#2f7d5c",
        }
        prediction_markers = {
            "positive": "^",
            "negative": "v",
            "neutral": "o",
        }
        prediction_words = {
            "positive": "UP",
            "negative": "DOWN",
            "neutral": "NEUTRAL",
        }
        if prediction.close_t is not None:
            prediction_color = label_colors.get(prediction.predicted_label, "#334155")
            axis.scatter(
                [mdates.date2num(event_timestamp)],
                [prediction.close_t],
                marker=prediction_markers.get(prediction.predicted_label, "o"),
                s=260,
                color=prediction_color,
                edgecolor="#111827",
                linewidth=1.2,
                zorder=5,
                label=f"prediction: {prediction_words.get(prediction.predicted_label, prediction.predicted_label)}",
            )
            axis.annotate(
                f"Prediction: {prediction_words.get(prediction.predicted_label, prediction.predicted_label)}",
                xy=(mdates.date2num(event_timestamp), prediction.close_t),
                xytext=(10, 18),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#111827",
                bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": prediction_color, "alpha": 0.92},
            )
        if prediction.close_t_plus_1 is not None and prediction.known_target is not None:
            actual_color = label_colors.get(prediction.known_target, "#334155")
            actual_matched = prediction.predicted_label == prediction.known_target
            actual_marker = "$✓$" if actual_matched else "$X$"
            actual_text = "Actual: match" if actual_matched else "Actual: miss"
            actual_box_color = "#dcfce7" if actual_matched else "#fee2e2"
            axis.scatter(
                [mdates.date2num(target_end_date)],
                [prediction.close_t_plus_1],
                marker=actual_marker,
                s=300,
                color=actual_color,
                edgecolor="#111827",
                linewidth=0.8,
                zorder=6,
                label=actual_text,
            )
            axis.annotate(
                f"{actual_text}: {prediction.known_target}",
                xy=(mdates.date2num(target_end_date), prediction.close_t_plus_1),
                xytext=(10, -24),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#111827",
                bbox={"boxstyle": "round,pad=0.25", "fc": actual_box_color, "ec": actual_color, "alpha": 0.92},
            )
        axis.set_title(
            f"{ticker.upper()} prior context through {cutoff_date:%Y-%m-%d}; "
            f"outcome through {target_end_date:%Y-%m-%d}"
        )
        axis.set_ylabel("Price")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        axis.grid(alpha=0.18)
        axis.legend(loc="upper left")
        figure.autofmt_xdate(rotation=35, ha="right")
        figure.tight_layout()
        return figure

    def _ticker_rows(self, ticker: str) -> pd.DataFrame:
        normalized_ticker = ticker.upper()
        return self.modeling_dataset.loc[
            self.modeling_dataset["ticker"].astype(str).str.upper().eq(normalized_ticker)
        ].copy()

    def _price_history(self, ticker: str) -> pd.DataFrame:
        prices = load_cached_prices(ticker.upper(), self.price_cache_dir)
        if prices.empty:
            raise ValueError(f"No cached prices found for ticker: {ticker}")
        return prices

    def _price_features(self, ticker: str) -> pd.DataFrame:
        prices = self._price_history(ticker)
        features = compute_price_feature_frame(prices, threshold=self.abnormal_threshold)
        features["event_date"] = pd.to_datetime(features["event_date"], errors="coerce").dt.normalize()
        return features.dropna(subset=PRICE_FEATURE_COLUMNS)

    def _price_feature_row(self, ticker: str, event_date: pd.Timestamp) -> pd.Series:
        features = self._price_features(ticker)
        matched = features.loc[features["event_date"].eq(event_date)]
        if matched.empty:
            available = features["event_date"].dropna().dt.strftime("%Y-%m-%d")
            raise ValueError(
                f"No complete prior-price features for {ticker.upper()} on {event_date:%Y-%m-%d}. "
                f"Available range: {available.min()} to {available.max()}."
            )
        return matched.iloc[0]

    def _build_input_frame(
        self,
        ticker: str,
        event_date: pd.Timestamp,
        message: str,
        feature_row: pd.Series,
    ) -> pd.DataFrame:
        text = message.strip() or f"${ticker.upper()} "
        row: dict[str, object] = {
            "ticker": ticker.upper(),
            "event_date": event_date,
            "original": text,
            "text_input": build_text_input(text, ticker.upper(), self.text_format),
        }
        for column in PRICE_FEATURE_COLUMNS:
            row[column] = feature_row[column]
        return pd.DataFrame([row])

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _optional_timestamp(value: object) -> pd.Timestamp | None:
        if pd.isna(value):
            return None
        return pd.Timestamp(value).normalize()
