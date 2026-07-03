"""Train classical MarketMood models."""

from __future__ import annotations

import argparse


def main() -> None:
    """CLI placeholder for classical-model training."""
    parser = argparse.ArgumentParser(description="Train classical models.")
    parser.add_argument("--config", default="config.yaml")
    parser.parse_args()
    raise NotImplementedError("Classical training will be implemented in Phase 4.")


if __name__ == "__main__":
    main()
