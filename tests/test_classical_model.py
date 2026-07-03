import pandas as pd

from marketmood.labels import LABEL_ORDER
from marketmood.models.classical import ClassicalLogisticModel, make_classical_models


def _tiny_modeling_frame() -> pd.DataFrame:
    rows = []
    labels = ["negative", "neutral", "positive", "negative", "neutral", "positive"]
    texts = [
        "bearish sell pressure",
        "flat mixed trading",
        "bullish breakout strength",
        "bearish downside pressure",
        "mixed sideways trading",
        "bullish upside breakout",
    ]
    for index, (label, text) in enumerate(zip(labels, texts, strict=True)):
        rows.append(
            {
                "target": label,
                "text_input": text,
                "emo_label": "oracle_emotion",
                "senti_label": "oracle_sentiment",
                "ret_1d": -0.01 + index * 0.004,
                "ret_3d": -0.02 + index * 0.006,
                "ret_5d": -0.03 + index * 0.01,
                "ret_10d": -0.04 + index * 0.012,
                "ret_20d": -0.05 + index * 0.014,
                "vol_5d": 0.02 + index * 0.001,
                "vol_10d": 0.025 + index * 0.001,
                "vol_20d": 0.03 + index * 0.001,
                "volume_z_20d": -1.0 + index * 0.4,
                "sma_5": 100 + index,
                "sma_20": 98 + index,
                "close_to_sma20": -0.02 + index * 0.01,
                "high_low_range": 0.02 + index * 0.002,
                "gap_return": -0.01 + index * 0.004,
                "range_position_20d": index / 5,
                "breakout_strength_20d": -0.02 + index * 0.01,
                "breakdown_strength_20d": 0.02 + index * 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_classical_model_variants_fit_predict_and_proba() -> None:
    frame = _tiny_modeling_frame()

    for feature_mode in ["price_only", "text_only", "text_price"]:
        model = ClassicalLogisticModel(feature_mode=feature_mode, max_features=20, max_iter=200)
        model.fit(frame)

        predictions = model.predict(frame)
        probabilities = model.predict_proba(frame)

        assert len(predictions) == len(frame)
        assert probabilities.columns.tolist() == LABEL_ORDER
        assert probabilities.shape == (len(frame), len(LABEL_ORDER))


def test_classical_models_do_not_use_oracle_annotation_labels() -> None:
    models = make_classical_models({"max_features": 20, "ngram_range": [1, 1], "max_iter": 200})

    for model in models:
        assert "emo_label" not in model.price_columns
        assert "senti_label" not in model.price_columns
        assert model.text_column == "text_input"
