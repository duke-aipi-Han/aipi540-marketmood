"""Classical machine-learning model components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marketmood.features import PRICE_FEATURE_COLUMNS
from marketmood.labels import LABEL_ORDER
from marketmood.metrics import classification_metrics
from marketmood.models.base import MarketMoodModel


ClassicalFeatureMode = Literal["price_only", "text_only", "text_price"]


PREDICTION_ID_COLUMNS = [
    "id",
    "split",
    "ticker",
    "post_date",
    "event_date",
    "original",
    "target",
]


@dataclass
class ClassicalLogisticModel(MarketMoodModel):
    """TF-IDF and price-feature logistic regression model."""

    feature_mode: ClassicalFeatureMode
    text_column: str = "text_input"
    price_columns: tuple[str, ...] = tuple(PRICE_FEATURE_COLUMNS)
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    max_iter: int = 2000
    class_weight: str | None = "balanced"

    def __post_init__(self) -> None:
        self.name = f"classical_{self.feature_mode}"
        self.pipeline = self._build_pipeline()

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit the configured classical model."""
        self._validate_columns(frame, require_target=True)
        self.pipeline.fit(frame, frame["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> list[str]:
        """Predict one abnormal-move class per row."""
        self._validate_columns(frame, require_target=False)
        return self.pipeline.predict(frame).tolist()

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities ordered as negative, neutral, positive."""
        self._validate_columns(frame, require_target=False)
        raw_probabilities = self.pipeline.predict_proba(frame)
        class_to_index = {
            label: index
            for index, label in enumerate(self.pipeline.named_steps["classifier"].classes_)
        }
        probabilities = pd.DataFrame(0.0, index=frame.index, columns=LABEL_ORDER)
        for label in LABEL_ORDER:
            if label in class_to_index:
                probabilities[label] = raw_probabilities[:, class_to_index[label]]
        return probabilities

    def save(self, model_dir: str | Path) -> Path:
        """Persist the model artifact and metadata."""
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"{self.feature_mode}.joblib"
        joblib.dump(self, artifact_path)
        metadata = {
            "model_name": self.name,
            "feature_mode": self.feature_mode,
            "text_column": self.text_column,
            "price_columns": list(self.price_columns),
            "labels": LABEL_ORDER,
        }
        with (output_dir / f"{self.feature_mode}_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        return artifact_path

    @classmethod
    def load(cls, artifact_path: str | Path) -> "ClassicalLogisticModel":
        """Load a persisted classical model artifact."""
        return joblib.load(artifact_path)

    def _build_pipeline(self) -> Pipeline:
        transformers = []

        if self.feature_mode in {"text_only", "text_price"}:
            transformers.append(
                (
                    "text",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        lowercase=True,
                        min_df=2,
                    ),
                    self.text_column,
                )
            )

        if self.feature_mode in {"price_only", "text_price"}:
            price_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("price", price_pipeline, list(self.price_columns)))

        if not transformers:
            raise ValueError(f"Unsupported classical feature mode: {self.feature_mode}")

        preprocessor = ColumnTransformer(transformers=transformers)
        classifier = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def _validate_columns(self, frame: pd.DataFrame, require_target: bool) -> None:
        required_columns: set[str] = set()
        if self.feature_mode in {"text_only", "text_price"}:
            required_columns.add(self.text_column)
        if self.feature_mode in {"price_only", "text_price"}:
            required_columns.update(self.price_columns)
        if require_target:
            required_columns.add("target")

        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing classical model columns: {sorted(missing)}")


def make_classical_models(config: dict[str, object]) -> list[ClassicalLogisticModel]:
    """Create all configured classical model variants."""
    ngram_range = tuple(config.get("ngram_range", [1, 2]))
    class_weight = config.get("class_weight", "balanced")
    if class_weight == "none":
        class_weight = None
    return [
        ClassicalLogisticModel(
            feature_mode=feature_mode,
            max_features=int(config.get("max_features", 50000)),
            ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
            max_iter=int(config.get("max_iter", 2000)),
            class_weight=class_weight,
        )
        for feature_mode in ["price_only", "text_only", "text_price"]
    ]


def build_prediction_frame(frame: pd.DataFrame, model: ClassicalLogisticModel) -> pd.DataFrame:
    """Create report-ready predictions for a fitted classical model."""
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame).add_prefix("prob_")
    available_columns = [column for column in PREDICTION_ID_COLUMNS if column in frame.columns]
    output = frame[available_columns].copy()
    output["predicted_label"] = predictions
    output = pd.concat([output.reset_index(drop=True), probabilities.reset_index(drop=True)], axis=1)
    return output.rename(columns={"target": "true_label"})


def train_classical_models(
    modeling_dataset_path: str | Path,
    model_dir: str | Path,
    metrics_path: str | Path,
    config: dict[str, object],
) -> dict[str, object]:
    """Train classical variants and save validation metrics."""
    frame = pd.read_csv(modeling_dataset_path)
    train_frame = frame.loc[frame["split"].eq("train")].copy()
    validation_frame = frame.loc[frame["split"].eq("validation")].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Training and validation splits are required for classical models.")

    metrics: dict[str, object] = {}
    best_model_name = ""
    best_macro_f1 = -1.0

    for model in make_classical_models(config):
        model.fit(train_frame)
        model.save(model_dir)
        predictions = model.predict(validation_frame)
        model_metrics = classification_metrics(
            validation_frame["target"].tolist(),
            predictions,
            labels=LABEL_ORDER,
        )
        model_metrics["model_name"] = model.name
        model_metrics["feature_mode"] = model.feature_mode
        model_metrics["split"] = "validation"
        model_metrics["n_rows"] = int(len(validation_frame))
        metrics[model.feature_mode] = model_metrics

        if model_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = model_metrics["macro_f1"]
            best_model_name = model.feature_mode

    metrics["best_model"] = best_model_name
    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def evaluate_classical_models(
    modeling_dataset_path: str | Path,
    model_dir: str | Path,
    predictions_dir: str | Path,
    metrics_path: str | Path,
    split: str = "test",
) -> dict[str, object]:
    """Evaluate all saved classical variants on one split."""
    frame = pd.read_csv(modeling_dataset_path)
    evaluation_frame = frame.loc[frame["split"].eq(split)].copy()
    if evaluation_frame.empty:
        raise ValueError(f"No rows found for split: {split}")

    metrics: dict[str, object] = {}
    prediction_output_dir = Path(predictions_dir)
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    for artifact_path in sorted(Path(model_dir).glob("*.joblib")):
        model = ClassicalLogisticModel.load(artifact_path)
        prediction_frame = build_prediction_frame(evaluation_frame, model)
        model_metrics = classification_metrics(
            prediction_frame["true_label"].tolist(),
            prediction_frame["predicted_label"].tolist(),
            labels=LABEL_ORDER,
        )
        model_metrics["model_name"] = model.name
        model_metrics["feature_mode"] = model.feature_mode
        model_metrics["split"] = split
        model_metrics["n_rows"] = int(len(prediction_frame))
        metrics[model.feature_mode] = model_metrics

        prediction_path = prediction_output_dir / f"classical_{model.feature_mode}_{split}_predictions.csv"
        prediction_frame.to_csv(prediction_path, index=False)

    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics
