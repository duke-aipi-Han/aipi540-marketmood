"""Gradio app for the MarketMood signal demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/marketmood-matplotlib")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure

from marketmood.config import load_config
from marketmood.inference.predict import MarketMoodInferenceService, SignalPrediction


def _default_selection(service: MarketMoodInferenceService) -> tuple[str, str, str, list[str]]:
    config = load_config(PROJECT_ROOT / "config.yaml")
    tickers = service.available_tickers()
    configured_ticker = str(config.values.get("app", {}).get("default_ticker", "AAPL")).upper()
    ticker = configured_ticker if configured_ticker in tickers else tickers[0]
    event_date = service.available_event_dates(ticker)[-1]
    posts = service.sample_posts(ticker, event_date)
    message = posts[0] if posts else f"${ticker} "
    return ticker, event_date, message, posts


def _context_frame(prediction: SignalPrediction) -> pd.DataFrame:
    rows = [
        ("ticker", prediction.ticker),
        ("event_date", prediction.event_date.strftime("%Y-%m-%d")),
        ("feature_cutoff_date", prediction.feature_cutoff_date.strftime("%Y-%m-%d")),
        ("ret_5d", f"{prediction.price_features['ret_5d']:.2%}"),
        ("ret_20d", f"{prediction.price_features['ret_20d']:.2%}"),
        ("vol_20d", f"{prediction.price_features['vol_20d']:.2%}"),
        ("range_position_20d", f"{prediction.price_features['range_position_20d']:.3f}"),
        ("breakout_strength_20d", f"{prediction.price_features['breakout_strength_20d']:.2%}"),
        ("breakdown_strength_20d", f"{prediction.price_features['breakdown_strength_20d']:.2%}"),
    ]
    if prediction.known_target is not None:
        rows.extend(
            [
                ("known_historical_target", prediction.known_target),
                ("known_future_return_1d", f"{prediction.known_future_return_1d:.2%}"),
                ("known_abnormal_score", f"{prediction.known_abnormal_score:.3f}"),
            ]
        )
    return pd.DataFrame(rows, columns=["field", "value"])


def _comparison_frame(prediction: SignalPrediction) -> pd.DataFrame:
    actual_label = prediction.known_target or "unavailable"
    match_value = (
        "yes"
        if prediction.known_target is not None and prediction.predicted_label == prediction.known_target
        else "no"
        if prediction.known_target is not None
        else "unavailable"
    )
    return pd.DataFrame(
        [
            {
                "event_date": prediction.event_date.strftime("%Y-%m-%d"),
                "model_prediction": prediction.predicted_label,
                "historical_actual": actual_label,
                "match": match_value,
                "feature_cutoff": prediction.feature_cutoff_date.strftime("%Y-%m-%d"),
                "actual_window_end": (
                    prediction.target_end_date.strftime("%Y-%m-%d")
                    if prediction.target_end_date is not None
                    else "unavailable"
                ),
                "next_day_return": (
                    f"{prediction.known_future_return_1d:.2%}"
                    if prediction.known_future_return_1d is not None
                    else "unavailable"
                ),
                "abnormal_score": (
                    f"{prediction.known_abnormal_score:.3f}"
                    if prediction.known_abnormal_score is not None
                    else "unavailable"
                ),
            }
        ]
    )


def build_app() -> gr.Blocks:
    """Create the Gradio Blocks application."""
    service = MarketMoodInferenceService.from_config(PROJECT_ROOT / "config.yaml", feature_mode="text_price")
    tickers = service.available_tickers()
    default_ticker, default_date, default_message, default_posts = _default_selection(service)
    default_prediction = service.predict(default_ticker, default_date, default_message)
    default_plot = service.plot_price_context(default_prediction)

    def set_ticker(ticker: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Textbox]:
        dates = service.available_event_dates(ticker)
        event_date = dates[-1]
        posts = service.sample_posts(ticker, event_date)
        message = posts[0] if posts else f"${ticker.upper()} "
        return (
            gr.update(choices=dates, value=event_date),
            gr.update(choices=posts, value=message if posts else None),
            gr.update(value=message),
        )

    def set_event_date(ticker: str, event_date: str) -> tuple[gr.Dropdown, gr.Textbox]:
        posts = service.sample_posts(ticker, event_date)
        message = posts[0] if posts else f"${ticker.upper()} "
        return (
            gr.update(choices=posts, value=message if posts else None),
            gr.update(value=message),
        )

    def set_sample_post(sample_post: str | None, current_message: str) -> str:
        return sample_post or current_message

    def run_prediction(
        ticker: str,
        event_date: str,
        message: str,
    ) -> tuple[dict[str, float], pd.DataFrame, Figure, pd.DataFrame]:
        prediction = service.predict(ticker, event_date, message)
        plot = service.plot_price_context(prediction)
        return (
            prediction.probabilities,
            _comparison_frame(prediction),
            plot,
            _context_frame(prediction),
        )

    with gr.Blocks(title="MarketMood Signal Demo") as demo:
        gr.Markdown(
            "# MarketMood Signal Demo\n"
            "Research demo for short-horizon abnormal-move classification from investor posts and recent price context."
        )
        with gr.Row():
            with gr.Column(scale=1):
                ticker_input = gr.Dropdown(
                    choices=tickers,
                    value=default_ticker,
                    label="Ticker",
                    interactive=True,
                )
                event_date_input = gr.Dropdown(
                    choices=service.available_event_dates(default_ticker),
                    value=default_date,
                    label="Event date",
                    interactive=True,
                )
                sample_post_input = gr.Dropdown(
                    choices=default_posts,
                    value=default_message if default_posts else None,
                    label="Historical sample post",
                    interactive=True,
                )
                message_input = gr.Textbox(
                    value=default_message,
                    label="Message",
                    lines=6,
                    max_lines=10,
                    interactive=True,
                )
                predict_button = gr.Button("Predict Signal", variant="primary")
            with gr.Column(scale=2):
                label_output = gr.Label(
                    value=default_prediction.probabilities,
                    label="Class probabilities",
                    num_top_classes=3,
                )
                comparison_output = gr.Dataframe(
                    value=_comparison_frame(default_prediction),
                    label="Prediction vs. historical actual",
                    interactive=False,
                )
                price_plot_output = gr.Plot(value=default_plot, label="Price context")
        with gr.Accordion("Market context", open=False):
            context_output = gr.Dataframe(
                value=_context_frame(default_prediction),
                interactive=False,
            )

        ticker_input.change(
            set_ticker,
            inputs=ticker_input,
            outputs=[event_date_input, sample_post_input, message_input],
        )
        event_date_input.change(
            set_event_date,
            inputs=[ticker_input, event_date_input],
            outputs=[sample_post_input, message_input],
        )
        sample_post_input.change(
            set_sample_post,
            inputs=[sample_post_input, message_input],
            outputs=message_input,
        )
        predict_button.click(
            run_prediction,
            inputs=[ticker_input, event_date_input, message_input],
            outputs=[label_output, comparison_output, price_plot_output, context_output],
        )

    return demo


def main() -> None:
    """Start the local Gradio app."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    port = int(config.values.get("app", {}).get("server_port", 8760))
    build_app().launch(server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    main()
