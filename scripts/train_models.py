"""Train configured MarketMood models."""

from __future__ import annotations

import sys
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketmood.config import load_config


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    """Train all currently implemented trainable models."""
    config = load_config(CONFIG_PATH)
    config.get_path("classical_model_dir").mkdir(parents=True, exist_ok=True)
    config.get_path("deep_fusion_model_dir").mkdir(parents=True, exist_ok=True)

    print("No trainable models are implemented yet.")
    print("The technical-analysis baseline is deterministic and is run by scripts/evaluate_models.py.")


if __name__ == "__main__":
    main()
