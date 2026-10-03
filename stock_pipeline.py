"""Stock data pipeline: fetch -> normalize -> indicators -> ML features.

Data-quality guarantees enforced here:
- Index normalized to UTC, sorted ascending, duplicate timestamps removed.
- OHLCV columns lowercased; yfinance MultiIndex columns flattened.
- Cache keys include ticker, interval, adjustment mode and the ticker's last
  bar date (derived from the UTC-normalized index, so the key follows the
  timezone change); TTL is only a fallback for same-bar refreshes.
- RSI uses Wilder smoothing; MACD keeps its 26+9 warmup window; warmup rows
  at the head stay NaN and are never backfilled.
- ML features are shift(1)-aligned to the decision time (no lookahead);
  features and labels are filtered jointly so they can never misalign;
  division-by-zero degrades to NaN-dropped rows and all-NaN feature columns
  are dropped explicitly.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]

OHLCV_FIELDS = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}
INDICATOR_COLUMNS = ["rsi", "macd", "macd_signal", "macd_hist", "sma_20", "sma_50"]


def _flatten_columns(columns):
    """Flatten yfinance MultiIndex columns, keeping the OHLCV field level."""
    if isinstance(columns, pd.MultiIndex):
        for level in range(columns.nlevels):
            values = [str(v) for v in columns.get_level_values(level)]
            if any(v.strip().lower() in OHLCV_FIELDS for v in values):
                return values
        return [str(v) for v in columns.get_level_values(0)]
    return [str(c) for c in columns]


def normalize_ohlcv(raw):
    """Return (df, stats) with lowercase columns and a sorted, deduped UTC index."""
    stats = {"duplicates_removed": 0, "empty_rows_removed": 0, "dropped_rows": 0}
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in _flatten_columns(df.columns)]
    df = df.loc[:, ~df.columns.duplicated()]

    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    df.index = index

    df = df.sort_index()
    dup_mask = df.index.duplicated(keep="last")
    stats["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    if "close" in df.columns:
        empty_mask = df["close"].isna()
    else:
        empty_mask = df.isna().all(axis=1)
    stats["empty_rows_removed"] = int(empty_mask.sum())
    df = df[~empty_mask]

    stats["dropped_rows"] = stats["duplicates_removed"] + stats["empty_rows_removed"]
    return df, stats



def _wilder_smooth(series, period):
    """Wilder smoothing: SMA seed over the first `period` valid values, then
    the recursive avg = (avg * (period - 1) + value) / period update."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    window = []
    avg = None
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        if avg is None:
            window.append(value)
            if len(window) == period:
                avg = sum(window) / period
                result[i] = avg
        else:
            avg = (avg * (period - 1) + value) / period
            result[i] = avg
    return pd.Series(result, index=series.index)


def compute_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal_period=9):
    """Add RSI (Wilder) and MACD columns; warmup head rows stay NaN."""
    out = df.copy()
    close = out["close"].astype(float)

    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()

    # RSI with Wilder smoothing (alpha = 1/period); min_periods keeps the
    # warmup window NaN instead of emitting partially-smoothed values.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_smooth(gain, rsi_period)
    avg_loss = _wilder_smooth(loss, rsi_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Explicit degenerate cases: gains with zero loss -> RSI 100;
    # a completely flat window (no gain, no loss) is undefined -> NaN.
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.eq(0), np.nan)
    out["rsi"] = rsi

    # MACD keeps the 26+9 warmup window: signal needs 9 valid MACD prints,
    # and MACD itself needs 26 closes, so the first 26+9-1 rows stay NaN.
    ema_fast = close.ewm(span=macd_fast, min_periods=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, min_periods=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal_period, min_periods=macd_signal_period, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    warmup_rows = {}
    valid_start = {}
    for col in INDICATOR_COLUMNS:
        first = out[col].first_valid_index()
        if first is None:
            warmup_rows[col] = len(out)
            valid_start[col] = None
        else:
            warmup_rows[col] = int(out.index.get_loc(first))
            valid_start[col] = first
    stats = {"warmup_rows": warmup_rows, "valid_start": valid_start}
    return out, stats


def build_ml_dataset(df):
    """Return (X, y, next_features, stats) with shift(1) decision-time alignment.

    Features at row t only use information available at t-1 (shift(1)); the
    label is the next-day return. Rows are dropped jointly on X and y so the
    two can never drift out of alignment. Infinite values (division by zero)
    degrade to NaN-dropped rows; all-NaN feature columns are dropped outright.
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)

    features = pd.DataFrame(index=df.index)
    features["ret_1d"] = close.pct_change()
    features["ret_5d"] = close.pct_change(5)
    features["ret_10d"] = close.pct_change(10)
    features["vol_10d"] = features["ret_1d"].rolling(10).std()
    features["close_vs_sma20"] = close / close.rolling(20).mean() - 1.0
    features["volume_chg_1d"] = volume.pct_change()
    if "rsi" in df:
        features["rsi"] = df["rsi"] / 100.0
    if "macd" in df:
        features["macd_pct"] = df["macd"] / close

    # Division by zero / overflow degrade to NaN; those rows drop out below.
    features = features.replace([np.inf, -np.inf], np.nan)

    # All-NaN feature columns carry no signal; drop them explicitly.
    all_nan_cols = features.columns[features.isna().all()].tolist()
    features = features.drop(columns=all_nan_cols)

    # shift(1): the decision at row t sees features computed through t-1.
    X = features.shift(1)
    y = close.pct_change().shift(-1)

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    # Features through the last available close, used to predict the next bar.
    last_row = features.iloc[-1:]
    next_features = last_row if last_row.notna().all(axis=1).iloc[0] else None

    stats = {
        "dropped_rows": int((~valid).sum()),
        "dropped_columns": all_nan_cols,
        "feature_columns": list(X.columns),
        "n_samples": int(len(X)),
    }
    return X, y, next_features, stats


def train_predictor(X, y, min_rows=60):
    """Train a RandomForest on the aligned dataset; None when data is short."""
    if X.shape[1] == 0 or len(X) < min_rows:
        return None
    from sklearn.ensemble import RandomForestRegressor

    split = max(int(len(X) * 0.8), 1)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    test_score = None
    if len(X) - split >= 5:
        test_score = float(model.score(X.iloc[split:], y.iloc[split:]))
    return {"model": model, "test_score": test_score, "n_train": split}



def synthetic_ohlcv(ticker, n=260):
    """Deterministic synthetic OHLCV used only as an explicit demo fallback."""
    seed = sum(ord(c) for c in ticker) % (2**32)
    rng = np.random.default_rng(seed)
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=n, freq="B")
    returns = rng.normal(0.0006, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


class QuoteCache:
    """Cache keyed by (ticker, interval, auto_adjust, last_bar_date).

    The last-bar date comes from the UTC-normalized index, so incremental
    quotes (a new bar) change the key and miss naturally; TTL is only a
    fallback for repeated refreshes within the same bar.
    """

    def __init__(self, ttl=300):
        self.ttl = ttl
        self._entries = {}
        self._last_bar = {}
        self.hits = 0
        self.misses = 0

    def key_for(self, ticker, interval, auto_adjust):
        last_bar = self._last_bar.get(ticker, "no-bar")
        return (ticker, interval, bool(auto_adjust), last_bar)

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None or time.time() - entry[0] > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        return entry[1]

    def put(self, ticker, interval, auto_adjust, df):
        self._last_bar[ticker] = df.index[-1].strftime("%Y-%m-%d")
        key = self.key_for(ticker, interval, auto_adjust)
        self._entries[key] = (time.time(), df)
        return key


class StockPipeline:
    """Fetch -> normalize -> indicators -> ML, per ticker, failure-isolated."""

    def __init__(self, tickers=None, period="1y", interval="1d", auto_adjust=True,
                 cache_ttl=300, synthetic_fallback=False):
        self.tickers = list(tickers or DEFAULT_TICKERS)
        self.period = period
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.synthetic_fallback = synthetic_fallback
        self._synthetic_tickers = set()
        self.cache = QuoteCache(ttl=cache_ttl)
        self.stats = {"cache_hits": 0, "cache_misses": 0, "failed_tickers": [],
                      "synthetic_tickers": [], "tickers": {}}

    def _fetch(self, ticker):
        if yf is None:
            return None
        try:
            df = yf.download(
                ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=self.auto_adjust,
                progress=False,
            )
        except Exception:
            return None
        if df is None or df.empty:
            return None
        return df

    def run(self):
        results = {}
        failed = []
        synthetic = []
        per_ticker = {}
        for ticker in self.tickers:
            key = self.cache.key_for(ticker, self.interval, self.auto_adjust)
            df = self.cache.get(key)
            if df is None:
                raw = self._fetch(ticker)
                source = "live"
                if raw is None:
                    if not self.synthetic_fallback:
                        failed.append(ticker)
                        continue
                    raw = synthetic_ohlcv(ticker)
                    source = "synthetic"
                    synthetic.append(ticker)
                    self._synthetic_tickers.add(ticker)
                df, norm_stats = normalize_ohlcv(raw)
                if df.empty:
                    failed.append(ticker)
                    continue
                self.cache.put(ticker, self.interval, self.auto_adjust, df)
            else:
                norm_stats = {"duplicates_removed": 0, "empty_rows_removed": 0, "dropped_rows": 0}

            data, ind_stats = compute_indicators(df)
            X, y, next_features, ml_stats = build_ml_dataset(data)

            prediction = None
            trained = train_predictor(X, y)
            if trained is not None and next_features is not None:
                pred_ret = float(trained["model"].predict(next_features)[0])
                last_close = float(data["close"].iloc[-1])
                prediction = {
                    "predicted_return": pred_ret,
                    "predicted_close": last_close * (1.0 + pred_ret),
                    "last_close": last_close,
                    "test_score": trained["test_score"],
                }

            results[ticker] = {"data": data, "prediction": prediction}
            per_ticker[ticker] = {
                "rows": int(len(data)),
                "last_bar": data.index[-1].strftime("%Y-%m-%d %H:%M %Z"),
                "dropped_rows": norm_stats["dropped_rows"],
                "warmup_rows": ind_stats["warmup_rows"],
                "valid_start": {
                    col: (ts.strftime("%Y-%m-%d") if ts is not None else None)
                    for col, ts in ind_stats["valid_start"].items()
                },
                "ml_dropped_rows": ml_stats["dropped_rows"],
                "ml_dropped_columns": ml_stats["dropped_columns"],
                "ml_samples": ml_stats["n_samples"],
                "prediction": prediction,
            }

        self.stats = {
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "failed_tickers": failed,
            "synthetic_tickers": sorted(self._synthetic_tickers),
            "tickers": per_ticker,
        }
        return results
