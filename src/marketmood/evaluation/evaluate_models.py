"""Evaluate trained MarketMood models."""

from __future__ import annotations

import argparse


def main() -> None:
    """CLI placeholder for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--config", default="config.yaml")
    parser.parse_args()
    raise NotImplementedError("Evaluation will be implemented in Phase 6.")


if __name__ == "__main__":
    main()
