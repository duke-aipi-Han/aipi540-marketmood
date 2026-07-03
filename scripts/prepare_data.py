"""Prepare cached prices and the modeling dataset."""

from __future__ import annotations

import sys
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketmood.config import load_config
from marketmood.data_load import load_stockemo_splits
from marketmood.features import build_modeling_dataset, print_class_distribution, save_modeling_outputs
from marketmood.prices import ensure_price_cache


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    """Prepare all local data artifacts with sensible defaults."""
    config = load_config(CONFIG_PATH)
    stockemo = load_stockemo_splits(
        config.get_path("train_csv"),
        config.get_path("val_csv"),
        config.get_path("test_csv"),
    )

    price_config = config.values.get("prices", {})
    statuses = ensure_price_cache(
        stockemo,
        config.get_path("price_cache_dir"),
        start_buffer_days=price_config.get("start_buffer_days", 90),
        end_buffer_days=price_config.get("end_buffer_days", 10),
        refresh=price_config.get("refresh_cache", False),
        ticker_aliases=price_config.get("ticker_aliases", {}),
    )
    empty_tickers = [ticker for ticker, status in statuses.items() if status == "0 rows"]
    print(f"Checked price cache for {len(statuses)} tickers.")
    if empty_tickers:
        print(f"Tickers with empty price history: {empty_tickers}")

    modeling, dropped_rows = build_modeling_dataset(
        stockemo,
        config.get_path("price_cache_dir"),
        threshold=float(config.values["labels"]["abnormal_threshold"]),
        text_format=config.values["features"].get("text_format", "raw"),
    )
    save_modeling_outputs(modeling, dropped_rows, config.get_path("modeling_dataset"))

    print(f"Saved {len(modeling)} modeling rows to {config.get_path('modeling_dataset')}")
    print(f"Saved {len(dropped_rows)} dropped-row records")
    print_class_distribution(modeling)


if __name__ == "__main__":
    main()
