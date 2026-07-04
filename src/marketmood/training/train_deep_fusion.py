"""Training and evaluation utilities for transformer fusion models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from marketmood.features import PRICE_FEATURE_COLUMNS
from marketmood.labels import LABEL_ORDER
from marketmood.metrics import classification_metrics
from marketmood.models.deep_fusion import (
    DeepFeatureMode,
    DeepFusionDataset,
    DeepModelConfig,
    ID_TO_LABEL,
    LABEL_TO_ID,
    PriceFeatureScaler,
    TransformerFusionClassifier,
    infer_text_hidden_size,
    load_deep_metadata,
    save_deep_artifact,
)


PREDICTION_ID_COLUMNS = [
    "id",
    "split",
    "ticker",
    "post_date",
    "event_date",
    "original",
    "target",
]


def select_torch_device(preference: str = "mps") -> torch.device:
    """Select the best available training device."""
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _class_weights(train_frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_frame["target"].value_counts()
    weights = []
    total = float(len(train_frame))
    for label in LABEL_ORDER:
        count = float(counts.get(label, 0))
        weights.append(total / (len(LABEL_ORDER) * max(count, 1.0)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _predict_batches(
    model: TransformerFusionClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    model.eval()
    predicted_labels: list[str] = []
    true_labels: list[str] = []
    probability_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            labels = batch.pop("labels", None)
            logits = model(**batch)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1).tolist()
            predicted_labels.extend(ID_TO_LABEL[index] for index in predictions)
            probability_chunks.append(probabilities)
            if labels is not None:
                true_labels.extend(ID_TO_LABEL[int(index)] for index in labels.detach().cpu().tolist())

    probabilities_frame = pd.DataFrame(
        torch.cat(probability_chunks, dim=0).numpy(),
        columns=[f"prob_{label}" for label in LABEL_ORDER],
    )
    return predicted_labels, probabilities_frame, true_labels


def _evaluate_loader(
    model: TransformerFusionClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    predictions, _, true_labels = _predict_batches(model, dataloader, device)
    return classification_metrics(true_labels, predictions, labels=LABEL_ORDER)


def _train_one_epoch(
    model: TransformerFusionClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        rows = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows

    return total_loss / max(total_rows, 1)


def _best_state_dict(model: TransformerFusionClassifier) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _make_model(text_encoder_name: str, feature_mode: DeepFeatureMode, config: dict[str, Any]):
    from transformers import AutoModel

    text_encoder = AutoModel.from_pretrained(text_encoder_name)
    return TransformerFusionClassifier(
        text_encoder=text_encoder,
        text_hidden_size=infer_text_hidden_size(text_encoder),
        feature_mode=feature_mode,
        price_feature_dim=len(PRICE_FEATURE_COLUMNS),
        price_hidden_dim=int(config.get("price_hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.2)),
        freeze_text_encoder=bool(config.get("freeze_text_encoder", True)),
        num_labels=len(LABEL_ORDER),
    )


def _make_tokenizer(text_encoder_name_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(text_encoder_name_or_path)


def _make_optimizer(
    model: TransformerFusionClassifier,
    encoder_learning_rate: float,
    head_learning_rate: float,
) -> torch.optim.Optimizer:
    encoder_parameters = [
        parameter
        for parameter in model.text_encoder.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("text_encoder.") and parameter.requires_grad
    ]
    parameter_groups = [{"params": head_parameters, "lr": head_learning_rate}]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_learning_rate})
    return torch.optim.AdamW(parameter_groups)


def train_deep_models(
    modeling_dataset_path: str | Path,
    model_dir: str | Path,
    metrics_path: str | Path,
    config: dict[str, Any],
    device_preference: str = "mps",
) -> dict[str, object]:
    """Train text-only and text-plus-price transformer models."""
    frame = pd.read_csv(modeling_dataset_path)
    train_frame = frame.loc[frame["split"].eq("train")].copy()
    validation_frame = frame.loc[frame["split"].eq("validation")].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Training and validation splits are required for deep models.")

    device = select_torch_device(device_preference)
    text_encoder_name = str(config.get("text_encoder", "distilbert-base-uncased"))
    max_length = int(config.get("max_length", 128))
    batch_size = int(config.get("batch_size", 16))
    epochs = int(config.get("epochs", 3))
    learning_rate = float(config.get("learning_rate", 2e-5))
    head_learning_rate = float(config.get("head_learning_rate", 1e-3))
    use_class_weights = bool(config.get("use_class_weights", True))
    feature_modes: list[DeepFeatureMode] = ["text_only", "text_price"]

    tokenizer = _make_tokenizer(text_encoder_name)
    metrics: dict[str, object] = {}
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature_mode in feature_modes:
        price_scaler = (
            PriceFeatureScaler.fit(train_frame, list(PRICE_FEATURE_COLUMNS))
            if feature_mode == "text_price"
            else None
        )
        train_dataset = DeepFusionDataset(
            train_frame,
            tokenizer=tokenizer,
            max_length=max_length,
            feature_mode=feature_mode,
            price_scaler=price_scaler,
            price_columns=list(PRICE_FEATURE_COLUMNS),
        )
        validation_dataset = DeepFusionDataset(
            validation_frame,
            tokenizer=tokenizer,
            max_length=max_length,
            feature_mode=feature_mode,
            price_scaler=price_scaler,
            price_columns=list(PRICE_FEATURE_COLUMNS),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        model = _make_model(text_encoder_name, feature_mode, config).to(device)
        optimizer = _make_optimizer(
            model,
            encoder_learning_rate=learning_rate,
            head_learning_rate=head_learning_rate,
        )
        loss_fn = nn.CrossEntropyLoss(
            weight=_class_weights(train_frame, device) if use_class_weights else None
        )

        best_macro_f1 = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, object] = {}
        history = []

        for epoch in range(epochs):
            train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            validation_metrics = _evaluate_loader(model, validation_loader, device)
            validation_metrics["train_loss"] = train_loss
            validation_metrics["epoch"] = epoch + 1
            history.append(validation_metrics)

            if float(validation_metrics["macro_f1"]) > best_macro_f1:
                best_macro_f1 = float(validation_metrics["macro_f1"])
                best_state = _best_state_dict(model)
                best_metrics = copy.deepcopy(validation_metrics)
            print(
                f"  {feature_mode} epoch {epoch + 1}/{epochs}: "
                f"loss={train_loss:.4f}, "
                f"validation macro F1={validation_metrics['macro_f1']:.3f}, "
                f"accuracy={validation_metrics['accuracy']:.3f}",
                flush=True,
            )

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for deep model: {feature_mode}")

        model.load_state_dict(best_state)
        artifact_config = DeepModelConfig(
            feature_mode=feature_mode,
            text_encoder=text_encoder_name,
            max_length=max_length,
            dropout=float(config.get("dropout", 0.2)),
            price_hidden_dim=int(config.get("price_hidden_dim", 64)),
            freeze_text_encoder=bool(config.get("freeze_text_encoder", True)),
            price_columns=list(PRICE_FEATURE_COLUMNS),
            labels=list(LABEL_ORDER),
        )
        artifact_dir = output_dir / feature_mode
        save_deep_artifact(
            artifact_dir=artifact_dir,
            model=model,
            tokenizer=tokenizer,
            config=artifact_config,
            price_scaler=price_scaler,
            validation_metrics=best_metrics,
        )

        best_metrics["model_name"] = f"deep_{feature_mode}"
        best_metrics["feature_mode"] = feature_mode
        best_metrics["split"] = "validation"
        best_metrics["n_rows"] = int(len(validation_frame))
        best_metrics["device"] = str(device)
        best_metrics["history"] = history
        metrics[feature_mode] = best_metrics

    best_model = max(
        feature_modes,
        key=lambda mode: float(metrics[mode]["macro_f1"]),
    )
    metrics["best_model"] = best_model

    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def load_deep_artifact(
    artifact_dir: str | Path,
    device: torch.device,
) -> tuple[TransformerFusionClassifier, Any, PriceFeatureScaler | None, DeepModelConfig]:
    from transformers import AutoModel

    artifact_path = Path(artifact_dir)
    metadata = load_deep_metadata(artifact_path)
    model_config = DeepModelConfig(**metadata["config"])
    tokenizer = _make_tokenizer(str(artifact_path / "tokenizer"))
    text_encoder = AutoModel.from_pretrained(model_config.text_encoder)
    model = TransformerFusionClassifier(
        text_encoder=text_encoder,
        text_hidden_size=infer_text_hidden_size(text_encoder),
        feature_mode=model_config.feature_mode,
        price_feature_dim=len(model_config.price_columns),
        price_hidden_dim=model_config.price_hidden_dim,
        dropout=model_config.dropout,
        freeze_text_encoder=model_config.freeze_text_encoder,
        num_labels=len(model_config.labels),
    )
    state_dict = torch.load(artifact_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    price_scaler = (
        PriceFeatureScaler.from_dict(metadata["price_scaler"])
        if metadata.get("price_scaler") is not None
        else None
    )
    return model, tokenizer, price_scaler, model_config


def build_deep_prediction_frame(
    frame: pd.DataFrame,
    model: TransformerFusionClassifier,
    tokenizer,
    model_config: DeepModelConfig,
    price_scaler: PriceFeatureScaler | None,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """Create report-ready predictions for a fitted deep model."""
    dataset = DeepFusionDataset(
        frame,
        tokenizer=tokenizer,
        max_length=model_config.max_length,
        feature_mode=model_config.feature_mode,
        price_scaler=price_scaler,
        price_columns=model_config.price_columns,
        include_labels="target" in frame.columns,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, probabilities, _ = _predict_batches(model, dataloader, device)
    available_columns = [column for column in PREDICTION_ID_COLUMNS if column in frame.columns]
    output = frame[available_columns].copy()
    output["predicted_label"] = predictions
    output = pd.concat([output.reset_index(drop=True), probabilities.reset_index(drop=True)], axis=1)
    return output.rename(columns={"target": "true_label"})


def evaluate_deep_models(
    modeling_dataset_path: str | Path,
    model_dir: str | Path,
    predictions_dir: str | Path,
    metrics_path: str | Path,
    config: dict[str, Any],
    split: str = "test",
    device_preference: str = "mps",
) -> dict[str, object]:
    """Evaluate saved deep variants on one split."""
    frame = pd.read_csv(modeling_dataset_path)
    evaluation_frame = frame.loc[frame["split"].eq(split)].copy()
    if evaluation_frame.empty:
        raise ValueError(f"No rows found for split: {split}")

    device = select_torch_device(device_preference)
    batch_size = int(config.get("batch_size", 16))
    metrics: dict[str, object] = {}
    prediction_output_dir = Path(predictions_dir)
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    artifact_dirs = [
        path for path in sorted(Path(model_dir).iterdir())
        if path.is_dir() and (path / "config.json").exists() and (path / "model.pt").exists()
    ]

    for artifact_dir in artifact_dirs:
        model, tokenizer, price_scaler, model_config = load_deep_artifact(artifact_dir, device)
        prediction_frame = build_deep_prediction_frame(
            evaluation_frame,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            price_scaler=price_scaler,
            device=device,
            batch_size=batch_size,
        )
        model_metrics = classification_metrics(
            prediction_frame["true_label"].tolist(),
            prediction_frame["predicted_label"].tolist(),
            labels=LABEL_ORDER,
        )
        model_metrics["model_name"] = f"deep_{model_config.feature_mode}"
        model_metrics["feature_mode"] = model_config.feature_mode
        model_metrics["split"] = split
        model_metrics["n_rows"] = int(len(prediction_frame))
        model_metrics["device"] = str(device)
        metrics[model_config.feature_mode] = model_metrics

        prediction_frame.to_csv(
            prediction_output_dir / f"deep_{model_config.feature_mode}_{split}_predictions.csv",
            index=False,
        )

    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics
