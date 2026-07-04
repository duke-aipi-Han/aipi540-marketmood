from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from marketmood.features import PRICE_FEATURE_COLUMNS
from marketmood.labels import LABEL_ORDER
from marketmood.models.deep_fusion import (
    DeepFusionDataset,
    PriceFeatureScaler,
    TransformerFusionClassifier,
)


class TinyTokenizer:
    def __call__(
        self,
        texts: list[str],
        max_length: int,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)
        for row_index, text in enumerate(texts):
            tokens = text.split()[:max_length]
            attention_mask[row_index, : len(tokens)] = 1
            for token_index, token in enumerate(tokens):
                input_ids[row_index, token_index] = (len(token) % 20) + 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _tiny_frame() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(LABEL_ORDER):
        row = {
            "text_input": f"{label} market setup",
            "target": label,
        }
        for feature_index, column in enumerate(PRICE_FEATURE_COLUMNS):
            row[column] = index + feature_index / 10
        rows.append(row)
    return pd.DataFrame(rows)


def test_deep_fusion_dataset_builds_text_and_price_tensors() -> None:
    frame = _tiny_frame()
    scaler = PriceFeatureScaler.fit(frame, list(PRICE_FEATURE_COLUMNS))
    dataset = DeepFusionDataset(
        frame,
        tokenizer=TinyTokenizer(),
        max_length=6,
        feature_mode="text_price",
        price_scaler=scaler,
    )

    item = dataset[0]

    assert len(dataset) == len(frame)
    assert item["input_ids"].shape == (6,)
    assert item["attention_mask"].shape == (6,)
    assert item["price_features"].shape == (len(PRICE_FEATURE_COLUMNS),)
    assert item["labels"].item() == 0


def test_transformer_fusion_classifier_forward_shapes() -> None:
    frame = _tiny_frame()
    scaler = PriceFeatureScaler.fit(frame, list(PRICE_FEATURE_COLUMNS))
    dataset = DeepFusionDataset(
        frame,
        tokenizer=TinyTokenizer(),
        max_length=5,
        feature_mode="text_price",
        price_scaler=scaler,
    )
    batch = {
        key: torch.stack([dataset[index][key] for index in range(len(dataset))])
        for key in ["input_ids", "attention_mask", "price_features"]
    }
    model = TransformerFusionClassifier(
        text_encoder=TinyEncoder(hidden_size=8),
        text_hidden_size=8,
        feature_mode="text_price",
        price_feature_dim=len(PRICE_FEATURE_COLUMNS),
        price_hidden_dim=4,
        dropout=0.1,
        num_labels=len(LABEL_ORDER),
    )

    logits = model(**batch)

    assert logits.shape == (len(frame), len(LABEL_ORDER))
