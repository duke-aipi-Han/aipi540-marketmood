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
