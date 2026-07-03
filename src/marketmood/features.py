"""Leakage-safe feature engineering for price and text inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from marketmood.labels import label_from_abnormal_score
from marketmood.prices import load_cached_prices
from marketmood.text_processing import build_text_input


PRICE_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "volume_z_20d",
    "sma_5",
    "sma_20",
    "close_to_sma20",
    "high_low_range",
    "gap_return",
    "range_position_20d",
    "breakout_strength_20d",
    "breakdown_strength_20d",
]


def _selected_close(prices: pd.DataFrame) -> pd.Series:
    """Prefer adjusted close when present; otherwise use close."""
    if "adj_close" in prices.columns and prices["adj_close"].notna().any():
        return prices["adj_close"].fillna(prices["close"])
    return prices["close"]


def compute_price_feature_frame(prices: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Create event-date rows with features through t-1 and targets from t to t+1."""
    if prices.empty:
        return pd.DataFrame()

    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    data = data.set_index("date")

    close = _selected_close(data)
    daily_return = close.pct_change()
    prev_close = close.shift(1)
    prev_volume = data["volume"].shift(1)
    volume_mean_20 = prev_volume.rolling(20).mean()
    volume_std_20 = prev_volume.rolling(20).std()
    sma_5 = prev_close.rolling(5).mean()
    sma_20 = prev_close.rolling(20).mean()
    prior_high_20 = close.shift(2).rolling(20).max()
    prior_low_20 = close.shift(2).rolling(20).min()
    prior_range_20 = prior_high_20 - prior_low_20
    rolling_vol_20d = daily_return.shift(1).rolling(20).std()

    features = pd.DataFrame(index=data.index)
    for horizon in [1, 3, 5, 10, 20]:
        features[f"ret_{horizon}d"] = close.shift(1) / close.shift(horizon + 1) - 1
    for window in [5, 10, 20]:
        features[f"vol_{window}d"] = daily_return.shift(1).rolling(window).std()

    features["volume_z_20d"] = (prev_volume - volume_mean_20) / volume_std_20
    features["sma_5"] = sma_5
    features["sma_20"] = sma_20
    features["close_to_sma20"] = prev_close / sma_20 - 1
    features["high_low_range"] = (data["high"].shift(1) - data["low"].shift(1)) / prev_close
    features["gap_return"] = data["open"].shift(1) / close.shift(2) - 1
    features["range_position_20d"] = (prev_close - prior_low_20) / prior_range_20
    features["breakout_strength_20d"] = prev_close / prior_high_20 - 1
    features["breakdown_strength_20d"] = prev_close / prior_low_20 - 1

    features["event_date"] = features.index
    features["feature_cutoff_date"] = features.index.to_series().shift(1)
    features["target_end_date"] = features.index.to_series().shift(-1)
    features["close_t"] = close
    features["close_t_plus_1"] = close.shift(-1)
    features["future_return_1d"] = close.shift(-1) / close - 1
    features["rolling_vol_20d"] = rolling_vol_20d
    features["abnormal_score"] = features["future_return_1d"] / rolling_vol_20d

    finite_scores = features["abnormal_score"].replace([np.inf, -np.inf], np.nan)
    features["abnormal_score"] = finite_scores
    features["target"] = pd.NA
    valid_score_mask = features["abnormal_score"].notna()
    features.loc[valid_score_mask, "target"] = features.loc[valid_score_mask, "abnormal_score"].map(
        lambda score: label_from_abnormal_score(float(score), threshold)
    )

    return features.reset_index(drop=True)


def align_to_event_date(post_date: pd.Timestamp, event_dates: pd.Series) -> pd.Timestamp | pd.NaT:
    """Map a post calendar date to the first available trading date on or after it."""
    if pd.isna(post_date) or event_dates.empty:
        return pd.NaT
    dates = pd.DatetimeIndex(pd.to_datetime(event_dates).sort_values().unique())
    position = dates.searchsorted(pd.Timestamp(post_date).normalize(), side="left")
    if position >= len(dates):
        return pd.NaT
    return dates[position]


def build_modeling_dataset(
    stockemo: pd.DataFrame,
    price_cache_dir: str | Path,
    threshold: float,
    text_format: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align StockEmotions posts to cached prices and return modeling rows plus drops."""
    rows: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    price_cache = Path(price_cache_dir)

    for ticker, ticker_posts in stockemo.groupby("ticker", sort=True):
        prices = load_cached_prices(ticker, price_cache)
        if prices.empty:
            for _, post in ticker_posts.iterrows():
                dropped.append({"id": post.get("id"), "ticker": ticker, "drop_reason": "missing_price_cache"})
            continue

        price_features = compute_price_feature_frame(prices, threshold=threshold)
        if price_features.empty:
            for _, post in ticker_posts.iterrows():
                dropped.append({"id": post.get("id"), "ticker": ticker, "drop_reason": "empty_price_features"})
            continue

        feature_lookup = price_features.set_index("event_date", drop=False)
        event_dates = feature_lookup.index.to_series()

        for _, post in ticker_posts.iterrows():
            event_date = align_to_event_date(post["date"], event_dates)
            if pd.isna(event_date):
                dropped.append({"id": post.get("id"), "ticker": ticker, "drop_reason": "no_event_date"})
                continue

            feature_row = feature_lookup.loc[event_date]
            required_columns = PRICE_FEATURE_COLUMNS + [
                "future_return_1d",
                "rolling_vol_20d",
                "abnormal_score",
                "target",
                "target_end_date",
                "feature_cutoff_date",
            ]
            if feature_row[required_columns].isna().any():
                dropped.append(
                    {
                        "id": post.get("id"),
                        "ticker": ticker,
                        "date": post.get("date"),
                        "event_date": event_date,
                        "drop_reason": "incomplete_features_or_target",
                    }
                )
                continue

            row = post.to_dict()
            row["post_date"] = post["date"]
            row["event_date"] = event_date
            row["feature_cutoff_date"] = feature_row["feature_cutoff_date"]
            row["target_end_date"] = feature_row["target_end_date"]
            row["text_raw"] = build_text_input(str(post["original"]), ticker, "raw")
            row["text_ticker_aware"] = build_text_input(str(post["original"]), ticker, "ticker_aware")
            row["text_ticker_masked"] = build_text_input(str(post["original"]), ticker, "ticker_masked")
            row["text_input"] = build_text_input(str(post["original"]), ticker, text_format)
            for column in PRICE_FEATURE_COLUMNS + [
                "close_t",
                "close_t_plus_1",
                "future_return_1d",
                "rolling_vol_20d",
                "abnormal_score",
                "target",
            ]:
                row[column] = feature_row[column]
            rows.append(row)

    modeling = pd.DataFrame(rows)
    dropped_rows = pd.DataFrame(dropped)
    if not modeling.empty:
        modeling = modeling.sort_values(["split", "event_date", "ticker", "id"]).reset_index(drop=True)
    return modeling, dropped_rows


def save_modeling_outputs(
    modeling: pd.DataFrame,
    dropped_rows: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Persist the modeling dataset and row-drop log."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(output, index=False)
    dropped_rows.to_csv(output.parent / "modeling_dataset_dropped_rows.csv", index=False)
    if not modeling.empty:
        distribution = pd.crosstab(modeling["split"], modeling["target"], margins=True)
        distribution.to_csv(output.parent / "modeling_dataset_class_distribution.csv")


def print_class_distribution(modeling: pd.DataFrame) -> None:
    """Print target distribution by split for a completed modeling dataset."""
    if modeling.empty:
        print("No modeling rows were generated.")
        return
    distribution = pd.crosstab(modeling["split"], modeling["target"], margins=True)
    print(distribution)
