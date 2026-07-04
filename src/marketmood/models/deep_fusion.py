"""Transformer and price-feature fusion model components."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from marketmood.features import PRICE_FEATURE_COLUMNS
from marketmood.labels import LABEL_ORDER


DeepFeatureMode = Literal["text_only", "text_price"]

LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class DeepModelConfig:
    """Serializable configuration for one deep model artifact."""

    feature_mode: DeepFeatureMode
    text_encoder: str
    max_length: int
    dropout: float
    price_hidden_dim: int
    freeze_text_encoder: bool
    price_columns: list[str]
    labels: list[str]


@dataclass(frozen=True)
class PriceFeatureScaler:
    """Simple train-split-only standardizer for engineered price features."""

    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "PriceFeatureScaler":
        """Fit feature means and standard deviations on one dataframe."""
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean().fillna(0.0)
        stds = numeric.std().replace(0.0, 1.0).fillna(1.0)
        return cls(
            means={column: float(means[column]) for column in columns},
            stds={column: float(stds[column]) for column in columns},
        )

    @classmethod
    def from_dict(cls, values: dict[str, dict[str, float]]) -> "PriceFeatureScaler":
        """Load a scaler from metadata values."""
        return cls(
            means={key: float(value) for key, value in values["means"].items()},
            stds={key: float(value) for key, value in values["stds"].items()},
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return JSON-serializable scaler values."""
        return {"means": self.means, "stds": self.stds}

    def transform(self, frame: pd.DataFrame, columns: list[str]) -> torch.Tensor:
        """Transform price columns to a float tensor."""
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce").copy()
        for column in columns:
            numeric[column] = (numeric[column].fillna(self.means[column]) - self.means[column]) / self.stds[column]
        return torch.tensor(numeric.to_numpy(dtype="float32"), dtype=torch.float32)


class DeepFusionDataset(Dataset):
    """Torch dataset for tokenized text, optional price features, and labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        max_length: int,
        feature_mode: DeepFeatureMode,
        price_scaler: PriceFeatureScaler | None = None,
        price_columns: list[str] | None = None,
        include_labels: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.feature_mode = feature_mode
        self.price_columns = price_columns or list(PRICE_FEATURE_COLUMNS)
        self.include_labels = include_labels

        texts = self.frame["text_input"].fillna("").astype(str).tolist()
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if feature_mode == "text_price":
            if price_scaler is None:
                raise ValueError("price_scaler is required for text_price datasets")
            self.price_features = price_scaler.transform(self.frame, self.price_columns)
        else:
            self.price_features = torch.empty((len(self.frame), 0), dtype=torch.float32)

        if include_labels:
            self.labels = torch.tensor(
                [LABEL_TO_ID[label] for label in self.frame["target"].tolist()],
                dtype=torch.long,
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["price_features"] = self.price_features[index]
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class TransformerFusionClassifier(nn.Module):
    """Transformer classifier with optional engineered-price fusion."""

    def __init__(
        self,
        text_encoder: nn.Module,
        text_hidden_size: int,
        feature_mode: DeepFeatureMode,
        price_feature_dim: int = len(PRICE_FEATURE_COLUMNS),
        price_hidden_dim: int = 64,
        dropout: float = 0.2,
        freeze_text_encoder: bool = False,
        num_labels: int = len(LABEL_ORDER),
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.feature_mode = feature_mode
        self.freeze_text_encoder = freeze_text_encoder
        self.dropout = nn.Dropout(dropout)
        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        fusion_dim = text_hidden_size
        if feature_mode == "text_price":
            self.price_mlp = nn.Sequential(
                nn.Linear(price_feature_dim, price_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(price_hidden_dim, price_hidden_dim),
                nn.ReLU(),
            )
            fusion_dim += price_hidden_dim
        else:
            self.price_mlp = None

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price_features: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return unnormalized class logits."""
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        if self.freeze_text_encoder:
            with torch.no_grad():
                try:
                    encoder_outputs = self.text_encoder(**encoder_inputs)
                except TypeError:
                    encoder_inputs.pop("token_type_ids", None)
                    encoder_outputs = self.text_encoder(**encoder_inputs)
        else:
            try:
                encoder_outputs = self.text_encoder(**encoder_inputs)
            except TypeError:
                encoder_inputs.pop("token_type_ids", None)
                encoder_outputs = self.text_encoder(**encoder_inputs)
        text_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        text_embedding = self.dropout(text_embedding)

        if self.feature_mode == "text_price":
            if price_features is None:
                raise ValueError("price_features are required for text_price models")
            price_embedding = self.price_mlp(price_features)
            text_embedding = torch.cat([text_embedding, price_embedding], dim=1)

        return self.classifier(text_embedding)


def infer_text_hidden_size(text_encoder: nn.Module) -> int:
    """Infer hidden size from a Hugging Face-style encoder module."""
    config = getattr(text_encoder, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not infer text encoder hidden size from encoder.config.hidden_size")
    return int(hidden_size)


def save_deep_artifact(
    artifact_dir: str | Path,
    model: TransformerFusionClassifier,
    tokenizer,
    config: DeepModelConfig,
    price_scaler: PriceFeatureScaler | None,
    validation_metrics: dict[str, object],
) -> None:
    """Persist weights, tokenizer, and metadata for one deep model variant."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "config": asdict(config),
        "price_scaler": price_scaler.to_dict() if price_scaler is not None else None,
        "validation_metrics": validation_metrics,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_deep_metadata(artifact_dir: str | Path) -> dict[str, object]:
    """Load saved deep model metadata."""
    with (Path(artifact_dir) / "config.json").open("r", encoding="utf-8") as file:
        return json.load(file)
