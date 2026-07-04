"""Evaluate configured MarketMood models."""

from __future__ import annotations

import sys
from pathlib import Path
import os

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketmood.baselines.technical_baseline import evaluate_technical_baseline
from marketmood.config import load_config
from marketmood.models.classical import evaluate_classical_models
from marketmood.training.train_deep_fusion import evaluate_deep_models


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    """Evaluate all currently implemented models with sensible defaults."""
    config = load_config(CONFIG_PATH)
    baseline_config = config.values["baseline"]
    summary_rows = []

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
    summary_rows.append(
        {
            "model": "technical_analysis_baseline",
            "feature_set": "price_rule",
            "split": "test",
            "accuracy": ta_metrics["accuracy"],
            "macro_f1": ta_metrics["macro_f1"],
            "weighted_f1": ta_metrics["weighted_f1"],
            "n_rows": ta_metrics["n_rows"],
        }
    )

    classical_model_dir = config.get_path("classical_model_dir")
    if any(classical_model_dir.glob("*.joblib")):
        classical_metrics = evaluate_classical_models(
            modeling_dataset_path=config.get_path("modeling_dataset"),
            model_dir=classical_model_dir,
            predictions_dir=config.get_path("predictions_dir"),
            metrics_path=config.get_path("metrics_dir") / "classical_metrics.json",
            split="test",
        )
        print("Classical models")
        for feature_mode, metrics in classical_metrics.items():
            summary_rows.append(
                {
                    "model": f"classical_{feature_mode}",
                    "feature_set": feature_mode,
                    "split": "test",
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "n_rows": metrics["n_rows"],
                }
            )
            print(
                f"  {feature_mode}: "
                f"rows={metrics['n_rows']}, "
                f"accuracy={metrics['accuracy']:.3f}, "
                f"macro F1={metrics['macro_f1']:.3f}, "
                f"weighted F1={metrics['weighted_f1']:.3f}"
            )
    else:
        print("Classical models")
        print("  no saved classical model artifacts found; run scripts/train_models.py first")

    deep_model_dir = config.get_path("deep_fusion_model_dir")
    has_deep_artifacts = deep_model_dir.exists() and any(
        (path / "model.pt").exists() for path in deep_model_dir.iterdir() if path.is_dir()
    )
    if has_deep_artifacts:
        deep_metrics = evaluate_deep_models(
            modeling_dataset_path=config.get_path("modeling_dataset"),
            model_dir=deep_model_dir,
            predictions_dir=config.get_path("predictions_dir"),
            metrics_path=config.get_path("metrics_dir") / "deep_fusion_metrics.json",
            config=config.values["deep_fusion"],
            split="test",
            device_preference=str(config.values["project"].get("device_preference", "mps")),
        )
        print("Deep models")
        for feature_mode, metrics in deep_metrics.items():
            summary_rows.append(
                {
                    "model": f"deep_{feature_mode}",
                    "feature_set": feature_mode,
                    "split": "test",
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "n_rows": metrics["n_rows"],
                }
            )
            print(
                f"  {feature_mode}: "
                f"rows={metrics['n_rows']}, "
                f"accuracy={metrics['accuracy']:.3f}, "
                f"macro F1={metrics['macro_f1']:.3f}, "
                f"weighted F1={metrics['weighted_f1']:.3f}, "
                f"device={metrics['device']}"
            )
    else:
        print("Deep models")
        print("  no saved deep model artifacts found; run scripts/train_models.py first")

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_path = config.get_path("metrics_dir") / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
