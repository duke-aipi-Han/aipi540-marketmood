import pandas as pd

from marketmood.baselines.technical_baseline import (
    TechnicalAnalysisBaseline,
    build_prediction_frame,
    predict_momentum_class,
)


def test_predict_momentum_class_uses_configured_threshold() -> None:
    assert predict_momentum_class(0.04, 0.04, threshold=0.75) == "positive"
    assert predict_momentum_class(-0.04, 0.04, threshold=0.75) == "negative"
    assert predict_momentum_class(0.01, 0.04, threshold=0.75) == "neutral"


def test_technical_analysis_baseline_predicts_breakout_setups() -> None:
    frame = pd.DataFrame(
        {
            "ret_5d": [0.04, -0.04, 0.04, -0.04, 0.01],
            "vol_20d": [0.04, 0.04, 0.04, 0.04, 0.04],
            "range_position_20d": [0.9, 0.1, 0.5, 0.5, 0.9],
            "breakout_strength_20d": [0.01, -0.10, 0.01, -0.10, 0.005],
            "breakdown_strength_20d": [0.10, -0.01, 0.10, -0.01, 0.10],
        }
    )
    model = TechnicalAnalysisBaseline(threshold=0.75)

    assert model.predict(frame) == ["positive", "negative", "neutral", "neutral", "neutral"]

    probabilities = model.predict_proba(frame)
    assert probabilities.columns.tolist() == ["negative", "neutral", "positive"]
    assert probabilities.sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert probabilities.loc[0, "positive"] == 1.0
    assert probabilities.loc[1, "negative"] == 1.0


def test_build_prediction_frame_includes_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "split": ["test"],
            "ticker": ["AAPL"],
            "post_date": ["2020-01-02"],
            "event_date": ["2020-01-02"],
            "original": ["$AAPL looking strong"],
            "target": ["positive"],
            "ret_5d": [0.04],
            "vol_20d": [0.04],
            "range_position_20d": [0.9],
            "breakout_strength_20d": [0.01],
            "breakdown_strength_20d": [0.10],
        }
    )
    prediction_frame = build_prediction_frame(frame, TechnicalAnalysisBaseline(threshold=0.75))

    assert prediction_frame.loc[0, "true_label"] == "positive"
    assert prediction_frame.loc[0, "predicted_label"] == "positive"
    assert prediction_frame.loc[0, "prob_positive"] == 1.0
    assert "breakout_score" in prediction_frame.columns
    assert "breakdown_score" in prediction_frame.columns
