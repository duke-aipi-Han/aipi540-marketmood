import pandas as pd

from marketmood.features import align_to_event_date, compute_price_feature_frame


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=35)
    close = pd.Series(range(100, 135), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": pd.Series(range(1000, 1035), dtype=float),
        }
    )


def test_compute_price_features_use_t_minus_1_and_target_uses_t_plus_1() -> None:
    features = compute_price_feature_frame(_synthetic_prices(), threshold=0.75)
    event_date = pd.Timestamp("2020-02-12")
    row = features.loc[features["event_date"].eq(event_date)].iloc[0]

    prior_close = 129.0
    close_t = 130.0
    close_t_plus_1 = 131.0

    assert row["feature_cutoff_date"] == pd.Timestamp("2020-02-11")
    assert row["target_end_date"] == pd.Timestamp("2020-02-13")
    assert row["ret_1d"] == prior_close / 128.0 - 1
    assert row["close_t"] == close_t
    assert row["close_t_plus_1"] == close_t_plus_1
    assert row["future_return_1d"] == close_t_plus_1 / close_t - 1


def test_align_to_event_date_uses_first_trading_date_on_or_after_post_date() -> None:
    event_dates = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]))

    assert align_to_event_date(pd.Timestamp("2020-01-04"), event_dates) == pd.Timestamp("2020-01-06")
    assert pd.isna(align_to_event_date(pd.Timestamp("2020-01-08"), event_dates))
