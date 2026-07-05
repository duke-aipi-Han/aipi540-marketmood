"""Simple signal backtesting for saved MarketMood predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from marketmood.evaluation.error_analysis import PREDICTION_FILES
from marketmood.evaluation.evaluate_models import MODEL_DISPLAY_NAMES, MODEL_ORDER
from marketmood.labels import LABEL_ORDER


BACKTEST_OUTPUT_COLUMNS = [
    "model",
    "display_name",
    "initial_capital",
    "final_value",
    "total_return",
    "max_drawdown",
    "sharpe_like",
    "n_trading_days",
    "n_trades",
    "win_rate",
    "average_trade_return",
    "average_daily_gross_exposure",
    "long_trades",
    "short_trades",
    "long_win_rate",
    "short_win_rate",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Settings for the simple one-day signal backtest."""

    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.05
    max_daily_gross_exposure: float = 1.0


def run_model_backtests(
    predictions_dir: str | Path,
    modeling_dataset_path: str | Path,
    output_dir: str | Path,
    figures_dir: str | Path,
    config: BacktestConfig | None = None,
) -> dict[str, Path]:
    """Run backtests for every saved model prediction file."""
    settings = config or BacktestConfig()
    prediction_root = Path(predictions_dir)
    backtest_dir = Path(output_dir)
    figure_dir = Path(figures_dir)
    backtest_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    context = _load_backtest_context(modeling_dataset_path)
    all_trades = []
    all_equity = []
    summary_rows = []

    for model_name in MODEL_ORDER:
        prediction_path = prediction_root / PREDICTION_FILES.get(model_name, "")
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        signals = build_daily_signals(predictions, context, model_name)
        trades, equity_curve, summary = simulate_signal_strategy(signals, model_name, settings)
        all_trades.append(trades)
        all_equity.append(equity_curve)
        summary_rows.append(summary)

    if not summary_rows:
        raise RuntimeError("No prediction files found for backtesting.")

    summary_frame = (
        pd.DataFrame(summary_rows)[BACKTEST_OUTPUT_COLUMNS]
        .sort_values("total_return", ascending=False)
        .reset_index(drop=True)
    )
    trades_frame = pd.concat(all_trades, ignore_index=True)
    equity_frame = pd.concat(all_equity, ignore_index=True)

    summary_path = backtest_dir / "backtest_summary.csv"
    trades_path = backtest_dir / "backtest_trades.csv"
    equity_path = backtest_dir / "backtest_equity_curves.csv"
    summary_frame.to_csv(summary_path, index=False)
    trades_frame.to_csv(trades_path, index=False)
    equity_frame.to_csv(equity_path, index=False)

    figure_paths = write_backtest_figures(summary_frame, equity_frame, figure_dir)
    paths = {
        "summary": summary_path,
        "trades": trades_path,
        "equity_curves": equity_path,
    }
    paths.update(figure_paths)
    return paths


def build_daily_signals(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Aggregate post-level predictions into one signal per ticker/date/model."""
    merged = predictions.merge(
        context,
        on="id",
        how="inner",
        suffixes=("", "_context"),
    )
    for column in ["ticker", "event_date", "target_end_date", "future_return_1d", "true_label"]:
        context_column = f"{column}_context"
        if context_column in merged.columns:
            merged[column] = merged[column].fillna(merged[context_column])

    probability_columns = [f"prob_{label}" for label in LABEL_ORDER]
    grouped = (
        merged.groupby(["ticker", "event_date"], as_index=False)
        .agg(
            {
                "target_end_date": "first",
                "future_return_1d": "first",
                "true_label": "first",
                "abnormal_score": "first",
                "id": "count",
                **{column: "mean" for column in probability_columns},
            }
        )
        .rename(columns={"id": "post_count"})
    )
    grouped["model"] = model_name
    grouped["display_name"] = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    probabilities = grouped[probability_columns]
    label_indexes = probabilities.to_numpy().argmax(axis=1)
    grouped["predicted_label"] = [LABEL_ORDER[index] for index in label_indexes]
    grouped["confidence"] = probabilities.max(axis=1)
    grouped["direction"] = grouped["predicted_label"].map({"negative": -1.0, "neutral": 0.0, "positive": 1.0})
    grouped["conviction"] = ((grouped["confidence"] - (1.0 / 3.0)) / (2.0 / 3.0)).clip(lower=0.0)
    grouped["event_date"] = pd.to_datetime(grouped["event_date"])
    grouped["target_end_date"] = pd.to_datetime(grouped["target_end_date"])
    return grouped.sort_values(["event_date", "ticker"]).reset_index(drop=True)


def simulate_signal_strategy(
    signals: pd.DataFrame,
    model_name: str,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Simulate a confidence-scaled one-day long/short strategy."""
    settings = config or BacktestConfig()
    equity = settings.initial_capital
    trades = []
    equity_rows = []

    for event_date, day_signals in signals.groupby("event_date", sort=True):
        day = day_signals.copy()
        day["position_pct_raw"] = day["direction"] * settings.max_position_pct * day["conviction"]
        gross_exposure = float(day["position_pct_raw"].abs().sum())
        scale = 1.0
        if gross_exposure > settings.max_daily_gross_exposure and gross_exposure > 0:
            scale = settings.max_daily_gross_exposure / gross_exposure
        day["position_pct"] = day["position_pct_raw"] * scale
        day["gross_position_pct"] = day["position_pct"].abs()
        day["trade_return"] = day["direction"] * day["future_return_1d"]
        day["pnl"] = equity * day["gross_position_pct"] * day["trade_return"]
        day["equity_start"] = equity
        day_pnl = float(day["pnl"].sum())
        day_return = day_pnl / equity if equity else 0.0
        equity_end = equity + day_pnl

        trades.append(day)
        equity_rows.append(
            {
                "model": model_name,
                "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "event_date": event_date,
                "equity_start": equity,
                "equity_end": equity_end,
                "daily_pnl": day_pnl,
                "daily_return": day_return,
                "gross_exposure": float(day["gross_position_pct"].sum()),
                "n_signals": int(len(day)),
                "n_trades": int((day["gross_position_pct"] > 0).sum()),
            }
        )
        equity = equity_end

    trades_frame = pd.concat(trades, ignore_index=True)
    equity_frame = pd.DataFrame(equity_rows)
    summary = summarize_backtest(trades_frame, equity_frame, settings)
    return trades_frame, equity_frame, summary


def summarize_backtest(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float | int | str]:
    """Calculate compact backtest comparison metrics."""
    model_name = str(equity_curve["model"].iloc[0])
    active_trades = trades[trades["gross_position_pct"] > 0].copy()
    long_trades = active_trades[active_trades["direction"] > 0]
    short_trades = active_trades[active_trades["direction"] < 0]
    final_value = float(equity_curve["equity_end"].iloc[-1])
    total_return = final_value / config.initial_capital - 1.0
    daily_returns = equity_curve["daily_return"].astype(float)
    daily_std = float(daily_returns.std(ddof=1))
    sharpe_like = 0.0 if daily_std == 0 else float((252**0.5) * daily_returns.mean() / daily_std)

    return {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "initial_capital": config.initial_capital,
        "final_value": final_value,
        "total_return": total_return,
        "max_drawdown": _max_drawdown(equity_curve["equity_end"]),
        "sharpe_like": sharpe_like,
        "n_trading_days": int(len(equity_curve)),
        "n_trades": int(len(active_trades)),
        "win_rate": _win_rate(active_trades),
        "average_trade_return": float(active_trades["trade_return"].mean()) if len(active_trades) else 0.0,
        "average_daily_gross_exposure": float(equity_curve["gross_exposure"].mean()),
        "long_trades": int(len(long_trades)),
        "short_trades": int(len(short_trades)),
        "long_win_rate": _win_rate(long_trades),
        "short_win_rate": _win_rate(short_trades),
    }


def write_backtest_figures(
    summary: pd.DataFrame,
    equity_curves: pd.DataFrame,
    figures_dir: str | Path,
) -> dict[str, Path]:
    """Write backtest comparison figures."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    paths = {
        "backtest_equity_curves": _plot_equity_curves(equity_curves, output_dir),
        "backtest_total_return": _plot_total_return(summary, output_dir),
        "backtest_risk_return": _plot_risk_return(summary, output_dir),
    }
    return paths


def _load_backtest_context(modeling_dataset_path: str | Path) -> pd.DataFrame:
    columns = [
        "id",
        "ticker",
        "event_date",
        "target_end_date",
        "future_return_1d",
        "abnormal_score",
        "target",
    ]
    context = pd.read_csv(modeling_dataset_path, usecols=columns).rename(columns={"target": "true_label"})
    return context


def _max_drawdown(equity_values: pd.Series) -> float:
    running_max = equity_values.cummax()
    drawdowns = equity_values / running_max - 1.0
    return float(drawdowns.min())


def _win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def _plot_equity_curves(equity_curves: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df = equity_curves.copy()
    plot_df["event_date"] = pd.to_datetime(plot_df["event_date"])
    sns.lineplot(data=plot_df, x="event_date", y="equity_end", hue="display_name", ax=ax)
    ax.set_title("Simple Signal Backtest Equity Curves")
    ax.set_xlabel("Event date")
    ax.set_ylabel("Portfolio value")
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "backtest_equity_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_total_return(summary: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = summary.sort_values("total_return", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, y="display_name", x="total_return", ax=ax, color="#4C78A8")
    ax.set_title("Backtest Total Return By Model")
    ax.set_xlabel("Total return")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for patch, value in zip(ax.patches, plot_df["total_return"]):
        x_position = patch.get_width()
        y_position = patch.get_y() + patch.get_height() / 2
        label_offset = 0.002 if x_position >= 0 else -0.002
        horizontal_alignment = "left" if x_position >= 0 else "right"
        ax.text(
            x_position + label_offset,
            y_position,
            f"{value:.1%}",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
        )
    fig.tight_layout()
    path = output_dir / "backtest_total_return.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_risk_return(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=summary,
        x="max_drawdown",
        y="total_return",
        hue="display_name",
        size="n_trades",
        sizes=(70, 220),
        ax=ax,
    )
    for _, row in summary.iterrows():
        ax.annotate(str(row["display_name"]), (row["max_drawdown"], row["total_return"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Backtest Return Versus Drawdown")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Total return")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="", loc="best")
    fig.tight_layout()
    path = output_dir / "backtest_return_vs_drawdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
