"""Generate report-ready evaluation plots and error-analysis tables."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/marketmood-matplotlib")

from marketmood.config import load_config
from marketmood.evaluation.error_analysis import write_error_analysis_artifacts
from marketmood.evaluation.evaluate_models import (
    load_metric_records,
    write_official_subgroup_metrics,
    write_metric_figures,
    write_metric_tables,
)


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    """Generate figures and example tables from saved evaluation outputs."""
    config = load_config(CONFIG_PATH)
    metrics_dir = config.get_path("metrics_dir")
    figures_dir = config.get_path("figures_dir")
    predictions_dir = config.get_path("predictions_dir")
    error_analysis_dir = config.get_path("error_analysis_dir")
    modeling_dataset = config.get_path("modeling_dataset")

    records = load_metric_records(metrics_dir)
    if not records:
        raise RuntimeError("No saved metric JSON files found. Run scripts/evaluate_models.py first.")

    table_paths = write_metric_tables(records, metrics_dir)
    subgroup_path = write_official_subgroup_metrics(
        predictions_path=predictions_dir / "deep_text_price_test_predictions.csv",
        modeling_dataset_path=modeling_dataset,
        metrics_dir=metrics_dir,
    )
    figure_paths = write_metric_figures(records, figures_dir)
    example_paths = write_error_analysis_artifacts(
        predictions_dir=predictions_dir,
        modeling_dataset_path=modeling_dataset,
        output_dir=error_analysis_dir,
    )

    print("Generated report metric tables")
    for name, path in table_paths.items():
        print(f"  {name}: {path}")
    print(f"  official_subgroup_metrics: {subgroup_path}")

    print("Generated report figures")
    for name, path in figure_paths.items():
        print(f"  {name}: {path}")

    print("Generated error-analysis artifacts")
    for name, path in example_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
