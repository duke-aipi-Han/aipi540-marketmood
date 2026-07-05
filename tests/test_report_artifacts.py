import json

import pandas as pd

from marketmood.evaluation.error_analysis import write_error_analysis_artifacts
from marketmood.evaluation.evaluate_models import load_metric_records, metrics_table, write_metric_tables


def test_load_metric_records_and_write_tables(tmp_path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    payload = {
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "weighted_f1": 0.45,
        "labels": ["negative", "neutral", "positive"],
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "classification_report": {
            "negative": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
            "neutral": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
            "positive": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
        },
        "model_name": "technical_analysis_baseline",
        "split": "test",
        "n_rows": 3,
    }
    (metrics_dir / "ta_baseline_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    records = load_metric_records(metrics_dir)
    table = metrics_table(records)
    paths = write_metric_tables(records, metrics_dir)

    assert records[0].model == "technical_analysis_baseline"
    assert table.loc[0, "display_name"] == "TA baseline"
    assert paths["model_comparison"].exists()
    assert paths["per_class_metrics"].exists()


def test_write_error_analysis_artifacts(tmp_path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    output_dir = tmp_path / "error_analysis"
    dataset_path = tmp_path / "modeling_dataset.csv"

    official = pd.DataFrame(
        [
            {
                "id": 1,
                "split": "test",
                "ticker": "TSLA",
                "post_date": "2020-01-02",
                "event_date": "2020-01-02",
                "original": "$TSLA down",
                "true_label": "negative",
                "predicted_label": "negative",
                "prob_negative": 0.9,
                "prob_neutral": 0.05,
                "prob_positive": 0.05,
            },
            {
                "id": 2,
                "split": "test",
                "ticker": "AAPL",
                "post_date": "2020-01-03",
                "event_date": "2020-01-03",
                "original": "$AAPL up",
                "true_label": "neutral",
                "predicted_label": "positive",
                "prob_negative": 0.1,
                "prob_neutral": 0.2,
                "prob_positive": 0.7,
            },
        ]
    )
    price_only = official.copy()
    price_only["predicted_label"] = ["positive", "neutral"]
    price_only[["prob_negative", "prob_neutral", "prob_positive"]] = [
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
    ]

    context = pd.DataFrame(
        [
            {
                "id": 1,
                "ticker": "TSLA",
                "event_date": "2020-01-02",
                "emo_label": "panic",
                "senti_label": "bearish",
                "future_return_1d": -0.05,
                "rolling_vol_20d": 0.02,
                "abnormal_score": -2.5,
                "range_position_20d": 0.1,
                "ret_5d": -0.03,
                "vol_20d": 0.02,
                "target": "negative",
            },
            {
                "id": 2,
                "ticker": "AAPL",
                "event_date": "2020-01-03",
                "emo_label": "optimism",
                "senti_label": "bullish",
                "future_return_1d": 0.001,
                "rolling_vol_20d": 0.02,
                "abnormal_score": 0.05,
                "range_position_20d": 0.5,
                "ret_5d": 0.01,
                "vol_20d": 0.02,
                "target": "neutral",
            },
        ]
    )

    official.to_csv(predictions_dir / "deep_text_price_test_predictions.csv", index=False)
    price_only.to_csv(predictions_dir / "classical_price_only_test_predictions.csv", index=False)
    context.to_csv(dataset_path, index=False)

    paths = write_error_analysis_artifacts(
        predictions_dir=predictions_dir,
        modeling_dataset_path=dataset_path,
        output_dir=output_dir,
        n_examples=2,
    )

    assert paths["text_price_wins"].exists()
    assert paths["price_only_wins"].exists()
    assert "deep text + price" in paths["example_summary"].read_text(encoding="utf-8")
