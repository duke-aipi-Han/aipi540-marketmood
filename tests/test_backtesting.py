import pandas as pd
import pytest

from marketmood.backtesting import BacktestConfig, build_daily_signals, simulate_signal_strategy


def test_build_daily_signals_aggregates_duplicate_posts() -> None:
    predictions = pd.DataFrame(
        [
            {
                "id": 1,
                "ticker": "AAPL",
                "event_date": "2020-01-02",
                "true_label": "positive",
                "predicted_label": "positive",
                "prob_negative": 0.1,
                "prob_neutral": 0.2,
                "prob_positive": 0.7,
            },
            {
                "id": 2,
                "ticker": "AAPL",
                "event_date": "2020-01-02",
                "true_label": "positive",
                "predicted_label": "neutral",
                "prob_negative": 0.1,
                "prob_neutral": 0.4,
                "prob_positive": 0.5,
            },
        ]
    )
    context = pd.DataFrame(
        [
            {
                "id": 1,
                "ticker": "AAPL",
                "event_date": "2020-01-02",
                "target_end_date": "2020-01-03",
                "future_return_1d": 0.02,
                "abnormal_score": 1.0,
                "true_label": "positive",
            },
            {
                "id": 2,
                "ticker": "AAPL",
                "event_date": "2020-01-02",
                "target_end_date": "2020-01-03",
                "future_return_1d": 0.02,
                "abnormal_score": 1.0,
                "true_label": "positive",
            },
        ]
    )

    signals = build_daily_signals(predictions, context, "deep_text_price")

    assert len(signals) == 1
    assert signals.loc[0, "post_count"] == 2
    assert signals.loc[0, "predicted_label"] == "positive"
    assert signals.loc[0, "confidence"] == 0.6


def test_simulate_signal_strategy_scales_position_by_confidence() -> None:
    signals = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "event_date": pd.Timestamp("2020-01-02"),
                "target_end_date": pd.Timestamp("2020-01-03"),
                "future_return_1d": 0.10,
                "true_label": "positive",
                "abnormal_score": 2.0,
                "post_count": 1,
                "prob_negative": 0.1,
                "prob_neutral": 0.2,
                "prob_positive": 0.7,
                "model": "deep_text_price",
                "display_name": "Deep text + price",
                "predicted_label": "positive",
                "confidence": 0.7,
                "direction": 1.0,
                "conviction": 0.55,
            }
        ]
    )

    trades, equity, summary = simulate_signal_strategy(
        signals,
        "deep_text_price",
        BacktestConfig(initial_capital=1_000.0, max_position_pct=0.1, max_daily_gross_exposure=1.0),
    )

    assert trades.loc[0, "position_pct"] == pytest.approx(0.055)
    assert trades.loc[0, "pnl"] == pytest.approx(5.5)
    assert equity.loc[0, "equity_end"] == pytest.approx(1005.5)
    assert summary["total_return"] == pytest.approx(0.0055)
