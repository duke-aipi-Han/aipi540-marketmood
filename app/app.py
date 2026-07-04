"""Gradio app for the MarketMood signal demo."""

from __future__ import annotations

import os
import sys
from html import escape
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


def _model_display_name(model_name: str) -> str:
    labels = {
        "deep_text_price": "Deep text + price",
        "deep_text_only": "Deep text-only",
        "classical_text_price": "Classical text + price",
        "classical_text_only": "Classical text-only",
        "classical_price_only": "Classical price-only",
    }
    return labels.get(model_name, model_name.replace("_", " ").title())


def _ordered_model_names(predictions: dict[str, SignalPrediction], official_model_name: str) -> list[str]:
    preferred_order = [
        official_model_name,
        "deep_text_only",
        "classical_text_price",
        "classical_text_only",
        "classical_price_only",
    ]
    ordered = [name for name in preferred_order if name in predictions]
    ordered.extend(name for name in predictions if name not in ordered)
    return ordered


def _comparison_frame(
    predictions: dict[str, SignalPrediction],
    official_model_name: str,
) -> str:
    headers = [
        "Model",
        "Prediction",
        "Confidence",
        "Historical actual",
        "Match",
        "Event date",
        "Actual window end",
        "Next-day return",
        "Abnormal score",
    ]
    rows = []
    for model_name in _ordered_model_names(predictions, official_model_name):
        prediction = predictions[model_name]
        actual_label = prediction.known_target or "unavailable"
        match_value = (
            "yes"
            if prediction.known_target is not None and prediction.predicted_label == prediction.known_target
            else "no"
            if prediction.known_target is not None
            else "unavailable"
        )
        rows.append(
            [
                _model_display_name(model_name),
                prediction.predicted_label,
                f"{max(prediction.probabilities.values()):.1%}",
                actual_label,
                match_value,
                prediction.event_date.strftime("%Y-%m-%d"),
                (
                    prediction.target_end_date.strftime("%Y-%m-%d")
                    if prediction.target_end_date is not None
                    else "unavailable"
                ),
                (
                    f"{prediction.known_future_return_1d:.2%}"
                    if prediction.known_future_return_1d is not None
                    else "unavailable"
                ),
                (
                    f"{prediction.known_abnormal_score:.3f}"
                    if prediction.known_abnormal_score is not None
                    else "unavailable"
                ),
                model_name == official_model_name,
            ]
        )

    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_html = ""
    for row in rows:
        is_official = row[-1]
        class_name = "official-row" if is_official else ""
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row[:-1])
        body_html += f"<tr class='{class_name}'>{cells}</tr>"
    return (
        "<style>"
        ".comparison-table{width:100%;border-collapse:collapse;font-size:0.92rem;color:inherit;background:transparent;}"
        ".comparison-table th,.comparison-table td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border-color-primary, currentColor);}"
        ".comparison-table th{font-weight:650;}"
        ".comparison-table .official-row td{font-weight:800;border-top:2px solid #f97316;border-bottom:2px solid #f97316;}"
        ".comparison-table .official-row td:first-child{border-left:5px solid #f97316;}"
        "</style>"
        "<table class='comparison-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def build_app() -> gr.Blocks:
    """Create the Gradio Blocks application."""
    service = MarketMoodInferenceService.from_config(PROJECT_ROOT / "config.yaml", feature_mode="text_price")
    tickers = service.available_tickers()
    default_ticker, default_date, default_message, default_posts = _default_selection(service)
    default_predictions = service.predict_all(default_ticker, default_date, default_message)
    default_prediction = default_predictions[service.official_model_name]
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
    ) -> tuple[dict[str, float], str, Figure, pd.DataFrame]:
        predictions = service.predict_all(ticker, event_date, message)
        official_prediction = predictions[service.official_model_name]
        plot = service.plot_price_context(official_prediction)
        return (
            official_prediction.probabilities,
            _comparison_frame(predictions, service.official_model_name),
            plot,
            _context_frame(official_prediction),
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
                comparison_output = gr.HTML(
                    value=_comparison_frame(default_predictions, service.official_model_name),
                    label="Model predictions vs. historical actual",
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
    is_space = bool(os.getenv("SPACE_ID"))
    server_name = "0.0.0.0" if is_space else "127.0.0.1"
    server_port = int(os.getenv("PORT", 7860 if is_space else port))
    build_app().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()
