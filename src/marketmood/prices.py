"""Historical price download and cache utilities."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CANONICAL_PRICE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def price_cache_path(ticker: str, cache_dir: str | Path) -> Path:
    """Return the local cache path for one ticker."""
    safe_ticker = ticker.upper().replace("/", "-")
    return Path(cache_dir) / f"{safe_ticker}.csv"


def standardize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance/cached OHLCV data to canonical columns."""
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Date" in data.columns:
        data = data.rename(columns={"Date": "date"})
    elif data.index.name in {"Date", "date"} or "date" not in data.columns:
        data = data.reset_index()
        if data.columns[0] != "date":
            data = data.rename(columns={data.columns[0]: "date"})

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename_map)
    data.columns = [str(column).strip().lower().replace(" ", "_") for column in data.columns]

    if "adj_close" not in data.columns:
        data["adj_close"] = data.get("close")

    for column in CANONICAL_PRICE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data = data[CANONICAL_PRICE_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    numeric_columns = [column for column in CANONICAL_PRICE_COLUMNS if column != "date"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    return data.reset_index(drop=True)


def load_cached_prices(ticker: str, cache_dir: str | Path) -> pd.DataFrame:
    """Load one ticker from the local price cache."""
    path = price_cache_path(ticker, cache_dir)
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)
    return standardize_price_frame(pd.read_csv(path))


def download_price_history(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ticker_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV prices from yfinance."""
    download_ticker = (ticker_aliases or {}).get(ticker.upper(), ticker)
    raw = yf.download(
        download_ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        actions=False,
        threads=False,
    )
    return standardize_price_frame(raw)


def get_price_history(
    ticker: str,
    cache_dir: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh: bool = False,
    ticker_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load cached prices or download and cache them if needed."""
    cache_path = price_cache_path(ticker, cache_dir)
    if cache_path.exists() and not refresh:
        return load_cached_prices(ticker, cache_dir)

    prices = download_price_history(ticker, start=start, end=end, ticker_aliases=ticker_aliases)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, index=False)
    return prices


def stockemo_price_window(
    stockemo: pd.DataFrame,
    start_buffer_days: int = 90,
    end_buffer_days: int = 10,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a yfinance download window covering StockEmotions plus buffers."""
    min_date = pd.to_datetime(stockemo["date"], errors="coerce").min()
    max_date = pd.to_datetime(stockemo["date"], errors="coerce").max()
    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError("Cannot infer price window because StockEmotions dates are missing.")
    start = (min_date - timedelta(days=start_buffer_days)).normalize()
    end = (max_date + timedelta(days=end_buffer_days)).normalize()
    return start, end


def ensure_price_cache(
    stockemo: pd.DataFrame,
    cache_dir: str | Path,
    start_buffer_days: int = 90,
    end_buffer_days: int = 10,
    refresh: bool = False,
    ticker_aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Download missing ticker prices and return ticker-level status messages."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    start, end = stockemo_price_window(stockemo, start_buffer_days, end_buffer_days)
    statuses: dict[str, str] = {}

    for ticker in sorted(stockemo["ticker"].dropna().astype(str).str.upper().unique()):
        try:
            prices = get_price_history(
                ticker,
                cache,
                start=start,
                end=end,
                refresh=refresh,
                ticker_aliases=ticker_aliases,
            )
        except Exception as exc:  # yfinance can raise several transport/data exceptions.
            statuses[ticker] = f"error: {exc}"
            continue
        statuses[ticker] = f"{len(prices)} rows"
    return statuses
