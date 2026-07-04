"""Train configured MarketMood models."""

from __future__ import annotations

import sys
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketmood.config import load_config
from marketmood.models.classical import train_classical_models
from marketmood.training.train_deep_fusion import train_deep_models


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    """Train all currently implemented trainable models."""
    config = load_config(CONFIG_PATH)
    config.get_path("classical_model_dir").mkdir(parents=True, exist_ok=True)
    config.get_path("deep_fusion_model_dir").mkdir(parents=True, exist_ok=True)

    classical_metrics = train_classical_models(
        modeling_dataset_path=config.get_path("modeling_dataset"),
        model_dir=config.get_path("classical_model_dir"),
        metrics_path=config.get_path("metrics_dir") / "classical_validation_metrics.json",
        config=config.values["classical"],
    )

    print("Classical models trained")
    for feature_mode, metrics in classical_metrics.items():
        if feature_mode == "best_model":
            continue
        print(
            f"  {feature_mode}: "
            f"validation macro F1={metrics['macro_f1']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f}"
        )
    print(f"Best validation model: {classical_metrics['best_model']}")

    deep_metrics = train_deep_models(
        modeling_dataset_path=config.get_path("modeling_dataset"),
        model_dir=config.get_path("deep_fusion_model_dir"),
        metrics_path=config.get_path("metrics_dir") / "deep_fusion_validation_metrics.json",
        config=config.values["deep_fusion"],
        device_preference=str(config.values["project"].get("device_preference", "mps")),
    )

    print("Deep models trained")
    for feature_mode, metrics in deep_metrics.items():
        if feature_mode == "best_model":
            continue
        print(
            f"  {feature_mode}: "
            f"validation macro F1={metrics['macro_f1']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"device={metrics['device']}"
        )
    print(f"Best validation deep model: {deep_metrics['best_model']}")


if __name__ == "__main__":
    main()
