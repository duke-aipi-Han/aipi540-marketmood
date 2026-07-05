"""Error-analysis helpers for report-ready examples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketmood.evaluation.evaluate_models import MODEL_DISPLAY_NAMES


PREDICTION_FILES = {
    "technical_analysis_baseline": "ta_baseline_test_predictions.csv",
    "classical_price_only": "classical_price_only_test_predictions.csv",
    "classical_text_only": "classical_text_only_test_predictions.csv",
    "classical_text_price": "classical_text_price_test_predictions.csv",
    "deep_text_only": "deep_text_only_test_predictions.csv",
    "deep_text_price": "deep_text_price_test_predictions.csv",
}


CONTEXT_COLUMNS = [
    "id",
    "ticker",
    "event_date",
    "emo_label",
    "senti_label",
    "future_return_1d",
    "rolling_vol_20d",
    "abnormal_score",
    "range_position_20d",
    "ret_5d",
    "vol_20d",
    "target",
]


def load_prediction_frame(
    predictions_dir: str | Path,
    model_name: str,
    modeling_dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load one model prediction file and attach optional modeling context."""
    path = Path(predictions_dir) / PREDICTION_FILES[model_name]
    frame = pd.read_csv(path)
    frame["model"] = model_name
    frame["display_name"] = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    frame["confidence"] = frame[[f"prob_{label}" for label in ["negative", "neutral", "positive"]]].max(axis=1)
    frame["is_correct"] = frame["true_label"] == frame["predicted_label"]

    if modeling_dataset_path is not None:
        context = pd.read_csv(modeling_dataset_path, usecols=lambda col: col in CONTEXT_COLUMNS)
        context = context.rename(columns={"target": "context_target"})
        merge_cols = [col for col in context.columns if col not in frame.columns or col == "id"]
        frame = frame.merge(context[merge_cols], on="id", how="left")
    return frame


def load_all_prediction_frames(
    predictions_dir: str | Path,
    modeling_dataset_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every available prediction file."""
    root = Path(predictions_dir)
    frames = {}
    for model_name, filename in PREDICTION_FILES.items():
        if (root / filename).exists():
            frames[model_name] = load_prediction_frame(root, model_name, modeling_dataset_path)
    return frames


def build_model_disagreement_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one row per test example with each model's prediction."""
    base_model = "deep_text_price" if "deep_text_price" in frames else next(iter(frames))
    base_columns = [
        "id",
        "ticker",
        "event_date",
        "original",
        "true_label",
        "future_return_1d",
        "abnormal_score",
        "senti_label",
        "emo_label",
    ]
    base = frames[base_model][[col for col in base_columns if col in frames[base_model].columns]].copy()

    for model_name, frame in frames.items():
        model_cols = ["id", "predicted_label", "confidence"]
        model_view = frame[model_cols].rename(
            columns={
                "predicted_label": f"{model_name}_prediction",
                "confidence": f"{model_name}_confidence",
            }
        )
        base = base.merge(model_view, on="id", how="left")
    return base


def write_error_analysis_artifacts(
    predictions_dir: str | Path,
    modeling_dataset_path: str | Path,
    output_dir: str | Path,
    official_model: str = "deep_text_price",
    comparison_model: str = "classical_price_only",
    n_examples: int = 12,
) -> dict[str, Path]:
    """Write concrete error-analysis CSV and Markdown artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames = load_all_prediction_frames(predictions_dir, modeling_dataset_path)
    official = frames[official_model].copy()
    official["likely_root_cause"] = official.apply(_likely_root_cause, axis=1)

    mistakes = (
        official[~official["is_correct"]]
        .sort_values(["confidence", "id"], ascending=[False, True])
        .head(n_examples)
    )
    correct = (
        official[official["is_correct"]]
        .sort_values(["confidence", "id"], ascending=[False, True])
        .head(n_examples)
    )

    disagreement = build_model_disagreement_table(frames)
    comparison = _build_pairwise_comparison(
        official=official,
        comparison=frames[comparison_model],
        official_model=official_model,
        comparison_model=comparison_model,
    )
    text_price_wins = comparison[comparison["official_wins"]].head(n_examples)
    price_only_wins = comparison[comparison["comparison_wins"]].head(n_examples)

    paths = {
        "official_high_confidence_mistakes": output_path / "official_high_confidence_mistakes.csv",
        "official_high_confidence_correct": output_path / "official_high_confidence_correct.csv",
        "model_disagreements": output_path / "model_disagreements.csv",
        "text_price_wins": output_path / "text_price_wins_over_price_only.csv",
        "price_only_wins": output_path / "price_only_wins_over_text_price.csv",
        "example_summary": output_path / "example_summary.md",
    }

    _write_compact_examples(mistakes, paths["official_high_confidence_mistakes"])
    _write_compact_examples(correct, paths["official_high_confidence_correct"])
    disagreement.to_csv(paths["model_disagreements"], index=False)
    _write_compact_examples(text_price_wins, paths["text_price_wins"])
    _write_compact_examples(price_only_wins, paths["price_only_wins"])
    _write_example_summary(
        mistakes=mistakes,
        text_price_wins=text_price_wins,
        price_only_wins=price_only_wins,
        path=paths["example_summary"],
    )
    return paths


def _build_pairwise_comparison(
    official: pd.DataFrame,
    comparison: pd.DataFrame,
    official_model: str,
    comparison_model: str,
) -> pd.DataFrame:
    columns = [
        "id",
        "ticker",
        "event_date",
        "original",
        "true_label",
        "predicted_label",
        "confidence",
        "future_return_1d",
        "abnormal_score",
        "senti_label",
        "emo_label",
        "likely_root_cause",
    ]
    official_view = official[[col for col in columns if col in official.columns]].rename(
        columns={
            "predicted_label": f"{official_model}_prediction",
            "confidence": f"{official_model}_confidence",
        }
    )
    comparison_view = comparison[["id", "predicted_label", "confidence"]].rename(
        columns={
            "predicted_label": f"{comparison_model}_prediction",
            "confidence": f"{comparison_model}_confidence",
        }
    )
    merged = official_view.merge(comparison_view, on="id", how="inner")
    merged["official_wins"] = (
        merged[f"{official_model}_prediction"].eq(merged["true_label"])
        & ~merged[f"{comparison_model}_prediction"].eq(merged["true_label"])
    )
    merged["comparison_wins"] = (
        merged[f"{comparison_model}_prediction"].eq(merged["true_label"])
        & ~merged[f"{official_model}_prediction"].eq(merged["true_label"])
    )
    merged["confidence_gap"] = (
        merged[f"{official_model}_confidence"] - merged[f"{comparison_model}_confidence"]
    ).abs()
    return merged.sort_values(
        ["confidence_gap", f"{official_model}_confidence", "id"],
        ascending=[False, False, True],
    )


def _write_compact_examples(frame: pd.DataFrame, path: Path) -> None:
    preferred_columns = [
        "id",
        "ticker",
        "event_date",
        "true_label",
        "predicted_label",
        "deep_text_price_prediction",
        "classical_price_only_prediction",
        "confidence",
        "deep_text_price_confidence",
        "classical_price_only_confidence",
        "future_return_1d",
        "abnormal_score",
        "senti_label",
        "emo_label",
        "likely_root_cause",
        "original",
    ]
    columns = [col for col in preferred_columns if col in frame.columns]
    frame[columns].to_csv(path, index=False)


def _write_example_summary(
    mistakes: pd.DataFrame,
    text_price_wins: pd.DataFrame,
    price_only_wins: pd.DataFrame,
    path: Path,
) -> None:
    sections = [
        "# Error Analysis Examples",
        "",
        "## High-Confidence Official Model Mistakes",
        _markdown_examples(mistakes.head(5)),
        "",
        "## Deep Text + Price Wins Over Price-Only",
        _markdown_examples(text_price_wins.head(3), prediction_column="deep_text_price_prediction"),
        "",
        "## Price-Only Wins Over Deep Text + Price",
        _markdown_examples(price_only_wins.head(3), prediction_column="classical_price_only_prediction"),
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def _markdown_examples(frame: pd.DataFrame, prediction_column: str = "predicted_label") -> str:
    if frame.empty:
        return "_No examples available._"

    lines = []
    for _, row in frame.iterrows():
        prediction = row.get(prediction_column, row.get("predicted_label", "unknown"))
        confidence = row.get("confidence", row.get("deep_text_price_confidence", float("nan")))
        model_comparison = _format_model_comparison(row)
        return_pct = _format_percent(row.get("future_return_1d"))
        abnormal = _format_number(row.get("abnormal_score"))
        text = str(row.get("original", "")).replace("\n", " ")
        if len(text) > 180:
            text = f"{text[:177]}..."
        lines.append(
            "- "
            f"{row.get('ticker')} on {row.get('event_date')}: "
            f"actual `{row.get('true_label')}`, predicted `{prediction}`, "
            f"confidence {_format_percent(confidence)}{model_comparison}, next-day return {return_pct}, "
            f"abnormal score {abnormal}. "
            f"Likely cause: {row.get('likely_root_cause', 'model disagreement or missing context')}. "
            f"Post: {text}"
        )
    return "\n".join(lines)


def _format_model_comparison(row: pd.Series) -> str:
    deep_prediction = row.get("deep_text_price_prediction")
    price_prediction = row.get("classical_price_only_prediction")
    if pd.isna(deep_prediction) or pd.isna(price_prediction):
        return ""
    deep_confidence = _format_percent(row.get("deep_text_price_confidence"))
    price_confidence = _format_percent(row.get("classical_price_only_confidence"))
    return (
        f" (deep text + price `{deep_prediction}` at {deep_confidence}; "
        f"price-only `{price_prediction}` at {price_confidence})"
    )


def _likely_root_cause(row: pd.Series) -> str:
    sentiment = str(row.get("senti_label", "")).lower()
    true_label = str(row.get("true_label", "")).lower()
    predicted = str(row.get("predicted_label", "")).lower()
    abnormal_score = abs(float(row.get("abnormal_score", 0.0) or 0.0))
    future_return = abs(float(row.get("future_return_1d", 0.0) or 0.0))
    text = str(row.get("original", ""))

    if (sentiment == "bullish" and true_label == "negative") or (
        sentiment == "bearish" and true_label == "positive"
    ):
        return "historical sentiment label conflicts with realized next-day move"
    if true_label == "neutral" and predicted in {"negative", "positive"}:
        return "model overreacted to directional cues in a neutral target window"
    if predicted == "neutral" and true_label in {"negative", "positive"}:
        return "abnormal move occurred despite a weak directional model signal"
    if abnormal_score >= 2.0 or future_return >= 0.04:
        return "large move likely reflects market/news context not present in the post"
    hype_markers = ["\U0001f602", "\U0001f923", "HAHA", "haha", "lol", "LOL", "\U0001f680", "\U0001f4b0"]
    if any(marker in text for marker in hype_markers):
        return "social-media shorthand, hype, or sarcasm may be hard to interpret"
    return "price context and message cues point in different directions"


def _format_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"
