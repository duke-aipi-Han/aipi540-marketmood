"""StockEmotions loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_STOCKEMO_COLUMNS = {
    "id",
    "date",
    "ticker",
    "emo_label",
    "senti_label",
    "original",
    "processed",
}


def load_stockemo_split(csv_path: str | Path) -> pd.DataFrame:
    """Load one StockEmotions split and validate its expected schema."""
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_STOCKEMO_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing StockEmotions columns: {sorted(missing)}")
    return frame


def load_stockemo_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> pd.DataFrame:
    """Load StockEmotions train/validation/test splits into one frame."""
    split_paths = {
        "train": train_csv,
        "validation": val_csv,
        "test": test_csv,
    }
    frames: list[pd.DataFrame] = []
    for split, path in split_paths.items():
        frame = load_stockemo_split(path).copy()
        frame["split"] = split
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
