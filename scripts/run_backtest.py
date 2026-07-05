"""Run the simple MarketMood signal backtest from saved predictions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/marketmood-matplotlib")

from marketmood.backtesting import BacktestConfig, run_model_backtests
from marketmood.config import load_config


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse optional backtest settings."""
    parser = argparse.ArgumentParser(description="Run simple one-day signal backtests.")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-position-pct", type=float, default=0.05)
    parser.add_argument("--max-daily-gross-exposure", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """Run backtests for all saved model prediction files."""
    args = parse_args()
    config = load_config(CONFIG_PATH)
    paths = run_model_backtests(
        predictions_dir=config.get_path("predictions_dir"),
        modeling_dataset_path=config.get_path("modeling_dataset"),
        output_dir=config.get_path("backtest_dir"),
        figures_dir=config.get_path("figures_dir"),
        config=BacktestConfig(
            initial_capital=args.initial_capital,
            max_position_pct=args.max_position_pct,
            max_daily_gross_exposure=args.max_daily_gross_exposure,
        ),
    )

    print("Generated backtest artifacts")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
