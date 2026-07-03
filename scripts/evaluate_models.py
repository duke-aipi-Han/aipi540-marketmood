"""Evaluate configured MarketMood models."""

from __future__ import annotations

import sys
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketmood.baselines.technical_baseline import evaluate_technical_baseline
from marketmood.config import load_config


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    """Evaluate all currently implemented models with sensible defaults."""
    config = load_config(CONFIG_PATH)
    baseline_config = config.values["baseline"]

    ta_metrics = evaluate_technical_baseline(
        modeling_dataset_path=config.get_path("modeling_dataset"),
        predictions_path=config.get_path("predictions_dir") / "ta_baseline_test_predictions.csv",
        metrics_path=config.get_path("metrics_dir") / "ta_baseline_metrics.json",
        threshold=float(baseline_config["technical_breakout_threshold"]),
        upper_range=float(baseline_config["breakout_upper_range"]),
        lower_range=float(baseline_config["breakout_lower_range"]),
        split="test",
    )

    print("Technical-analysis baseline")
    print(f"  rows:        {ta_metrics['n_rows']}")
    print(f"  accuracy:    {ta_metrics['accuracy']:.3f}")
    print(f"  macro F1:    {ta_metrics['macro_f1']:.3f}")
    print(f"  weighted F1: {ta_metrics['weighted_f1']:.3f}")


if __name__ == "__main__":
    main()
