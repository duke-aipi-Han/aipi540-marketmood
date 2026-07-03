"""Historical price download and cache utilities."""

from __future__ import annotations

import argparse


def main() -> None:
    """CLI placeholder for the price-cache pipeline."""
    parser = argparse.ArgumentParser(description="Download/cache ticker prices.")
    parser.add_argument("--config", default="config.yaml")
    parser.parse_args()
    raise NotImplementedError("Price cache pipeline will be implemented in Phase 2.")


if __name__ == "__main__":
    main()
