"""Report-artifact utilities built from saved model metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

from marketmood.labels import LABEL_ORDER


MODEL_DISPLAY_NAMES = {
    "technical_analysis_baseline": "TA baseline",
    "classical_price_only": "Classical price-only",
    "classical_text_only": "Classical text-only",
    "classical_text_price": "Classical text + price",
    "deep_text_only": "Deep text-only",
    "deep_text_price": "Deep text + price",
}


MODEL_ORDER = [
    "deep_text_price",
    "classical_price_only",
    "classical_text_price",
    "technical_analysis_baseline",
    "classical_text_only",
    "deep_text_only",
]


@dataclass(frozen=True)
class MetricRecord:
    """Metric bundle for one evaluated model."""

    model: str
    display_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]


def load_metric_records(metrics_dir: str | Path) -> list[MetricRecord]:
    """Load all saved test metric JSON files into a stable display order."""
    metrics_path = Path(metrics_dir)
    records: dict[str, MetricRecord] = {}

    ta_path = metrics_path / "ta_baseline_metrics.json"
    if ta_path.exists():
        payload = _read_json(ta_path)
        records[str(payload["model_name"])] = _record_from_payload(payload)

    for path in [metrics_path / "classical_metrics.json", metrics_path / "deep_fusion_metrics.json"]:
        if not path.exists():
            continue
        payload = _read_json(path)
        for model_payload in payload.values():
            records[str(model_payload["model_name"])] = _record_from_payload(model_payload)

    ordered = [records[name] for name in MODEL_ORDER if name in records]
    ordered.extend(record for name, record in records.items() if name not in MODEL_ORDER)
    return ordered


def metrics_table(records: list[MetricRecord]) -> pd.DataFrame:
    """Return the compact model-comparison table used by plots and reports."""
    return pd.DataFrame(
        [
            {
                "model": record.model,
                "display_name": record.display_name,
                "accuracy": record.accuracy,
                "macro_f1": record.macro_f1,
                "weighted_f1": record.weighted_f1,
            }
            for record in records
        ]
    )


def per_class_f1_table(records: list[MetricRecord]) -> pd.DataFrame:
    """Return per-class F1 scores for each model."""
    rows = []
    for record in records:
        report = record.classification_report
        for label in LABEL_ORDER:
            label_metrics = report.get(label, {})
            rows.append(
                {
                    "model": record.model,
                    "display_name": record.display_name,
                    "class": label,
                    "f1": float(label_metrics.get("f1-score", 0.0)),
                    "precision": float(label_metrics.get("precision", 0.0)),
                    "recall": float(label_metrics.get("recall", 0.0)),
                    "support": float(label_metrics.get("support", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def write_metric_tables(records: list[MetricRecord], metrics_dir: str | Path) -> dict[str, Path]:
    """Write report-friendly CSV metric tables."""
    output_dir = Path(metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "report_model_comparison.csv"
    class_path = output_dir / "report_per_class_metrics.csv"
    metrics_table(records).to_csv(model_path, index=False)
    per_class_f1_table(records).to_csv(class_path, index=False)
    return {"model_comparison": model_path, "per_class_metrics": class_path}


def write_official_subgroup_metrics(
    predictions_path: str | Path,
    modeling_dataset_path: str | Path,
    metrics_dir: str | Path,
    minimum_group_size: int = 10,
) -> Path:
    """Write simple subgroup metrics for the official model predictions."""
    predictions = pd.read_csv(predictions_path)
    context = pd.read_csv(
        modeling_dataset_path,
        usecols=[
            "id",
            "ticker",
            "emo_label",
            "senti_label",
            "original",
            "vol_20d",
            "abnormal_score",
        ],
    )
    frame = predictions.merge(context, on="id", how="left", suffixes=("", "_context"))
    frame["ticker"] = frame["ticker"].fillna(frame.get("ticker_context"))
    frame["post_length_bucket"] = pd.qcut(
        frame["original"].astype(str).str.len(),
        q=4,
        labels=["short", "medium_short", "medium_long", "long"],
        duplicates="drop",
    )
    frame["volatility_regime"] = pd.qcut(
        frame["vol_20d"],
        q=3,
        labels=["low_volatility", "medium_volatility", "high_volatility"],
        duplicates="drop",
    )
    frame["contains_non_ascii"] = frame["original"].astype(str).str.contains(r"[^\x00-\x7F]", regex=True)

    rows = []
    for column in [
        "ticker",
        "senti_label",
        "emo_label",
        "volatility_regime",
        "post_length_bucket",
        "contains_non_ascii",
    ]:
        rows.extend(_subgroup_rows(frame, column, minimum_group_size))

    output_dir = Path(metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report_subgroup_metrics.csv"
    pd.DataFrame(rows).sort_values(["group_type", "macro_f1"], ascending=[True, False]).to_csv(path, index=False)
    return path


def write_metric_figures(records: list[MetricRecord], figures_dir: str | Path) -> dict[str, Path]:
    """Write report-ready model-comparison and confusion-matrix figures."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    paths = {
        "model_comparison": _plot_model_comparison(records, output_dir),
        "per_class_f1": _plot_per_class_f1(records, output_dir),
        "ablation": _plot_ablation(records, output_dir),
        "confusion_matrix_grid": _plot_confusion_matrix_grid(records, output_dir),
    }

    for record in records:
        key = f"confusion_matrix_{record.model}"
        paths[key] = _plot_single_confusion_matrix(record, output_dir)
    return paths


def _subgroup_rows(frame: pd.DataFrame, column: str, minimum_group_size: int) -> list[dict[str, object]]:
    rows = []
    for value, group in frame.dropna(subset=[column]).groupby(column, observed=True):
        if len(group) < minimum_group_size:
            continue
        rows.append(
            {
                "group_type": column,
                "group_value": str(value),
                "n_rows": int(len(group)),
                "accuracy": float(accuracy_score(group["true_label"], group["predicted_label"])),
                "macro_f1": float(
                    f1_score(
                        group["true_label"],
                        group["predicted_label"],
                        labels=LABEL_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "mean_abs_abnormal_score": float(group["abnormal_score"].abs().mean()),
            }
        )
    return rows


def _record_from_payload(payload: dict[str, object]) -> MetricRecord:
    model = str(payload["model_name"])
    return MetricRecord(
        model=model,
        display_name=MODEL_DISPLAY_NAMES.get(model, model.replace("_", " ").title()),
        accuracy=float(payload["accuracy"]),
        macro_f1=float(payload["macro_f1"]),
        weighted_f1=float(payload["weighted_f1"]),
        confusion_matrix=list(payload["confusion_matrix"]),
        classification_report=dict(payload["classification_report"]),
    )


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _plot_model_comparison(records: list[MetricRecord], output_dir: Path) -> Path:
    table = metrics_table(records).melt(
        id_vars=["display_name"],
        value_vars=["macro_f1", "weighted_f1", "accuracy"],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "accuracy": "Accuracy",
    }
    table["metric"] = table["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=table, y="display_name", x="score", hue="metric", ax=ax, palette="Set2")
    ax.set_title("Model Comparison On Test Split")
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.set_xlim(0, 0.75)
    ax.legend(title="")
    _annotate_grouped_bars(ax)
    fig.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_per_class_f1(records: list[MetricRecord], output_dir: Path) -> Path:
    table = per_class_f1_table(records)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=table, y="display_name", x="f1", hue="class", ax=ax, palette="Set1")
    ax.set_title("Per-Class F1 On Test Split")
    ax.set_xlabel("F1")
    ax.set_ylabel("")
    ax.set_xlim(0, 0.8)
    ax.legend(title="Class")
    fig.tight_layout()
    path = output_dir / "per_class_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_ablation(records: list[MetricRecord], output_dir: Path) -> Path:
    table = metrics_table(records)
    ablation_rows = []
    for row in table.to_dict("records"):
        model = str(row["model"])
        if model == "technical_analysis_baseline":
            family = "Rule"
            feature_set = "Price rule"
        elif model.startswith("classical_"):
            family = "Classical"
            feature_set = _feature_set_label(model.removeprefix("classical_"))
        elif model.startswith("deep_"):
            family = "Deep"
            feature_set = _feature_set_label(model.removeprefix("deep_"))
        else:
            continue
        ablation_rows.append(
            {
                "family": family,
                "feature_set": feature_set,
                "macro_f1": float(row["macro_f1"]),
            }
        )

    plot_df = pd.DataFrame(ablation_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="family",
        y="macro_f1",
        hue="feature_set",
        ax=ax,
        palette="Set2",
    )
    ax.set_title("Feature Ablation: Price, Text, And Fusion")
    ax.set_xlabel("")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 0.7)
    ax.legend(title="Input")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    path = output_dir / "feature_ablation_macro_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_confusion_matrix_grid(records: list[MetricRecord], output_dir: Path) -> Path:
    n_cols = 3
    n_rows = (len(records) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, record in zip(axes_list, records):
        _draw_confusion_matrix(ax, record)
    for ax in axes_list[len(records) :]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices On Test Split", y=1.01, fontsize=14)
    fig.tight_layout()
    path = output_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_single_confusion_matrix(record: MetricRecord, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    _draw_confusion_matrix(ax, record)
    fig.tight_layout()
    path = output_dir / f"confusion_matrix_{record.model}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _draw_confusion_matrix(ax: plt.Axes, record: MetricRecord) -> None:
    matrix = pd.DataFrame(record.confusion_matrix, index=LABEL_ORDER, columns=LABEL_ORDER)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(record.display_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def _feature_set_label(feature_set: str) -> str:
    return {
        "price_only": "Price only",
        "text_only": "Text only",
        "text_price": "Text + price",
    }.get(feature_set, feature_set.replace("_", " ").title())


def _annotate_grouped_bars(ax: plt.Axes) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
