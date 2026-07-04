from pathlib import Path

import pytest

from marketmood.labels import LABEL_ORDER
from marketmood.inference import MarketMoodInferenceService


def test_inference_service_predicts_from_message_and_prior_price_context() -> None:
    if not Path("models/classical/text_price.joblib").exists():
        pytest.skip("classical text_price artifact is not available")

    service = MarketMoodInferenceService.from_config("config.yaml", feature_mode="text_price")
    ticker = "AAPL" if "AAPL" in service.available_tickers() else service.available_tickers()[0]
    event_date = service.available_event_dates(ticker)[0]
    message = service.default_message(ticker, event_date)

    prediction = service.predict(ticker, event_date, message)

    assert prediction.predicted_label in LABEL_ORDER
    assert list(prediction.probabilities) == LABEL_ORDER
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.feature_cutoff_date < prediction.event_date
    assert "emo_label" not in prediction.price_features
    assert "senti_label" not in prediction.price_features
