"""Leakage-safe feature engineering for price and text inputs."""

from __future__ import annotations

import argparse


def main() -> None:
    """CLI placeholder for modeling-dataset feature generation."""
    parser = argparse.ArgumentParser(description="Build modeling features.")
    parser.add_argument("--config", default="config.yaml")
    parser.parse_args()
    raise NotImplementedError("Feature generation will be implemented in Phase 2.")


if __name__ == "__main__":
    main()
