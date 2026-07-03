import pytest

from marketmood.labels import label_from_abnormal_score


def test_label_from_abnormal_score_uses_strict_thresholds() -> None:
    assert label_from_abnormal_score(0.76, 0.75) == "positive"
    assert label_from_abnormal_score(-0.76, 0.75) == "negative"
    assert label_from_abnormal_score(0.75, 0.75) == "neutral"
    assert label_from_abnormal_score(-0.75, 0.75) == "neutral"


def test_label_from_abnormal_score_rejects_nan() -> None:
    with pytest.raises(ValueError):
        label_from_abnormal_score(float("nan"), 0.75)
