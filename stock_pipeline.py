"""
stock_pipeline
==============

End-to-end, *correct* data pipeline for the stock dashboard::

    fetch -> normalize (UTC / lowercase / flat columns) -> indicators
          -> ML dataset (no look-ahead) -> model -> prediction

Design rules enforced in this module
------------------------------------
* All frames are sorted ascending, deduplicated and tz-normalized to UTC.
* OHLCV column names are lower-case ``open/high/low/close/volume`` and any
  ``pandas.MultiIndex`` returned by yfinance is flattened.
* RSI uses Wilder's smoothing (RMA); MACD keeps the 26 + 9 warm-up window.
  Insufficient warm-up -> leading NaN rows, never back/forward filled.
* ML features are shifted by one row so a decision at the close of day *t*
  only uses information released on or before day *t*.  Features and the
  label are aligned on the same index before rows are dropped.
* Divisions by zero and all-NaN feature columns degrade to NaN / dropped
  columns rather than raising.
* One ticker failing can never take the whole page down.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# ttl is only a back-stop; cache freshness is driven by the last bar date
# that is part of the key.  12h keeps on-disk garbage from living forever.
DEFAULT_CACHE_TTL_SECONDS = 12 * 60 * 60

# adjustment modes accepted by the pipeline / cache key
ADJUST_RAW = "raw"
ADJUST_QFQ = "qfq"
ADJUST_HFQ = "hfq"


class PipelineError(RuntimeError):
    """Raised when an individual ticker cannot be processed."""


# ---------------------------------------------------------------------------
# Fetching & normalization
# ---------------------------------------------------------------------------

def _flatten_columns(columns: pd.Index) -> pd.Index:
    """Flatten a possible MultiIndex to a plain single-level Index."""
    if isinstance(columns, pd.MultiIndex):
        flat = []
        for parts in columns:
            name = next((str(p) for p in parts if p is not None and str(p) != ""), "")
            flat.append(name)
        return pd.Index(flat)
    return columns


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw yfinance frame.

* MultiIndex columns are flattened.
    * OHLCV columns become lower-case ``open/high/low/close/volume``.
    * The index becomes a tz-aware ``DatetimeIndex`` in UTC, sorted ascending,
      with duplicate timestamps removed (last row wins).
    """
    if data is None:
        raise PipelineError("no data returned")

    df = data.copy()
    df.columns = _flatten_columns(df.columns)

    # yfinance sometimes uses capitalised or spaced column labels.
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ("open", "high", "low", "close", "volume"):
            rename_map[str(col)] = key
        elif key == "adj_close" or key == "adjclose" or key == "adjusted_close":
            rename_map[str(col)] = "adj_close"
        elif key in ("dividends", "dividend"):
            rename_map[str(col)] = "dividends"
        elif key in ("stock_splits", "splits", "split"):
            rename_map[str(col)] = "stock_splits"
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise PipelineError(f"missing required OHLCV columns: {missing}")

    # Datetime index, UTC normalized.
    idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    df.index = idx
    df = df[~df.index.isna()]
    df = df.sort_index(kind="mergesort")

    # Deduplicate identical timestamps (keep the most recently reported row).
    df = df[~df.index.duplicated(keep="last")]

    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        raise PipelineError("empty frame after normalization")
    return df


def fetch_raw_ohlcv(
    ticker: str,
    interval: str = "1d",
    period: str = "2y",
    adjust: str = ADJUST_QFQ,
) -> pd.DataFrame:
    """Download one ticker from Yahoo Finance. Raises on failure/empty."""
    if adjust not in (ADJUST_RAW, ADJUST_QFQ, ADJUST_HFQ):
        raise ValueError(f"unsupported adjust mode: {adjust!r}")

    auto_adjust = adjust == ADJUST_QFQ
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise PipelineError(f"yfinance returned no rows for {ticker}")

    df = normalize_ohlcv(data)

    if adjust == ADJUST_HFQ:
        # Back-adjusted ("hfq") prices: raw * cumulative split multiplier.
        if "stock_splits" in df.columns and df["stock_splits"].fillna(0).ne(0).any():
            splits = df["stock_splits"].replace(0, np.nan).fillna(1.0)
            split_mult = splits.iloc[::-1].cumprod().iloc[::-1]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * split_mult

    if adjust == ADJUST_RAW and "adj_close" in df.columns:
        # Keep raw OHLC untouched; adj_close stays available for reference.
        pass

    return df


def last_bar_date(df: pd.DataFrame) -> str:
    """UTC calendar date (YYYY-MM-DD) of the last bar in a normalized frame."""
    return pd.Timestamp(df.index.max()).tz_convert("UTC").date().isoformat()


# ---------------------------------------------------------------------------
# Cache: key = ticker|interval|adjust|last bar date (UTC), ttl only back-stop
# ---------------------------------------------------------------------------

def make_cache_key(
    ticker: str,
    interval: str,
    adjust: str,
    last_date: str,
) -> str:
    raw = f"{ticker.upper()}|{interval}|{adjust}|{last_date}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{ticker.upper()}_{interval}_{adjust}_{last_date}_{digest[:12]}"


class DataCache:
    """Tiny on-disk pickle cache keyed by the ticker's UTC last-bar date."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] = ".cache",
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        with self._lock:
            if not path.exists():
                return None
            # ttl is *only* a back-stop for abandoned/partial files.
            if time.time() - path.stat().st_mtime > self.ttl_seconds:
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
            try:
                with path.open("rb") as fh:
                    return pickle.load(fh)
            except (pickle.PickleError, EOFError, OSError):
                return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        with self._lock:
            with tmp.open("wb") as fh:
                pickle.dump(value, fh)
            os.replace(tmp, path)

    def load_ohlcv(
        self,
        ticker: str,
        interval: str,
        adjust: str,
        fetcher: Callable[[], pd.DataFrame],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Resolve OHLCV for one ticker.

        Because the key embeds the (UTC) last bar date it must be computed
        from fresh data first; an incremental new bar therefore changes the
        key and can never hit yesterday's cached payload.
        """
        meta: dict[str, Any] = {
            "ticker": ticker.upper(),
            "interval": interval,
            "adjust": adjust,
            "cache_hit": False,
            "stale_fallback": False,
            "cache_key": None,
            "last_bar_date": None,
            "rows": 0,
        }

        try:
            # 1) Fetch first so the key always carries the *current* last bar.
            fresh = normalize_ohlcv(fetcher())
        except Exception as fetch_exc:
            # Network/rate-limit failure: fall back to the newest payload we
            # previously stored for this ticker/interval/adjust.  It is
            # flagged as a stale fallback so the UI never presents it as live.
            stale = self.latest_for(ticker, interval, adjust)
            if stale is None:
                raise
            key, frame = stale
            meta["cache_hit"] = True
            meta["stale_fallback"] = True
            meta["cache_key"] = key
            meta["last_bar_date"] = last_bar_date(frame)
            meta["rows"] = len(frame)
            meta["fetch_error"] = f"{type(fetch_exc).__name__}: {fetch_exc}"
            return frame, meta

        meta["last_bar_date"] = last_bar_date(fresh)
        key = make_cache_key(ticker, interval, adjust, meta["last_bar_date"])
        meta["cache_key"] = key

        # 2) Same key -> payload reusable across processes/restarts.
        cached = self.get(key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            meta["cache_hit"] = True
            meta["rows"] = len(cached)
            return cached, meta

        self.set(key, fresh)
        meta["rows"] = len(fresh)
        return fresh, meta

    def latest_for(
        self, ticker: str, interval: str, adjust: str
    ) -> tuple[str, pd.DataFrame] | None:
        """Newest non-expired cached frame for a ticker/interval/adjust."""
        prefix = f"{ticker.upper()}_{interval}_{adjust}_"
        candidates = [
            p for p in self.cache_dir.glob(f"{prefix}*.pkl")
            if time.time() - p.stat().st_mtime <= self.ttl_seconds
        ]
        if not candidates:
            return None
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with path.open("rb") as fh:
                frame = pickle.load(fh)
        except (pickle.PickleError, EOFError, OSError):
            return None
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        return path.stem, frame
# ---------------------------------------------------------------------------
# Technical indicators (Wilder RSI, 26+9 warm-up MACD, no back-fill)
# ---------------------------------------------------------------------------

def _rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothed moving average (RMA).

    Seeded with a simple average of the first ``period`` values, then
    ``avg = (prev * (period - 1) + value) / period``.  The first
    ``period - 1`` rows are NaN (insufficient warm-up) and are *not* filled.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=series.index)

    seed = np.nanmean(values[:period])
    out[period - 1] = seed
    running = seed
    for i in range(period, n):
        value = values[i]
        running = (running * (period - 1) + value) / period
        out[i] = running
    return pd.Series(out, index=series.index)


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI with Wilder smoothing.

    First ``period`` rows are NaN.  A zero average loss with positive gain
    degrades to 100; zero gain AND zero loss (flat line) degrades to 50;
    zero gain with positive loss degrades to 0.  NaN prices stay NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    flat = avg_gain.eq(0.0) & avg_loss.eq(0.0)
    all_gain = avg_gain.gt(0.0) & avg_loss.eq(0.0)
    rsi = rsi.mask(flat, 50.0)
    rsi = rsi.mask(all_gain, 100.0)
    # avg_gain == 0 & avg_loss > 0 -> 100 - 100/(1+0) = 0 already correct.
    return rsi.rename("rsi")


def _sma_seeded_ema(values: np.ndarray, span: int) -> np.ndarray:
    """
    EMA seeded with an SMA of the first ``span`` values, then recursive
    smoothing with ``alpha = 2 / (span + 1)``.  Leading ``span - 1`` rows
    stay NaN (classic warm-up convention used by MACD).
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < span:
        return out
    alpha = 2.0 / (span + 1.0)
    seed = np.nanmean(values[:span])
    out[span - 1] = seed
    running = seed
    for i in range(span, n):
        running = alpha * values[i] + (1.0 - alpha) * running
        out[i] = running
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD using SMA-seeded EMAs.

    Warm-up rows (head only, never back-filled):
    * ``ema_fast`` valid from bar ``fast`` (first ``fast - 1`` NaN)
    * ``ema_slow`` / ``macd`` valid from bar ``slow`` (``slow - 1`` NaN)
    * ``macd_signal`` / ``macd_histogram`` valid from bar
      ``slow + signal - 1`` (``slow + signal - 2`` NaN rows)
    """
    values = close.to_numpy(dtype=float)
    ema_fast = pd.Series(_sma_seeded_ema(values, fast), index=close.index)
    ema_slow = pd.Series(_sma_seeded_ema(values, slow), index=close.index)
    macd_line = ema_fast - ema_slow

    # Signal is an EMA of the MACD line, seeded over its first 9 *valid*
    # bars (the 26-bar warm-up must be excluded from the seed window).
    signal_values = np.full(len(values), np.nan)
    valid_start = slow - 1
    valid = macd_line.iloc[valid_start:].to_numpy(dtype=float)
    if len(valid) >= signal:
        signal_values[valid_start:] = _sma_seeded_ema(valid, signal)
    signal_line = pd.Series(signal_values, index=close.index)

    return pd.DataFrame(
        {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_line - signal_line,
        }
    )


def calculate_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Append the full indicator set to a normalized OHLCV frame.

    Returns the enriched frame and a meta dict describing the warm-up:
    ``valid_starts`` maps indicator -> first valid UTC date (or None),
    ``warmup_rows`` maps indicator -> number of leading NaN rows.
    """
    out = df.copy()
    close = out["close"]

    # Moving averages (rolling -> leading NaN by construction).
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_200"] = close.rolling(200).mean()

    # MACD family with proper warm-up.
    out = pd.concat([out, macd(close)], axis=1)

    # Wilder RSI.
    out["rsi"] = wilder_rsi(close, 14)

    # Bollinger Bands.
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_middle"] = bb_mid
    out["bb_upper"] = bb_mid + 2.0 * bb_std
    out["bb_lower"] = bb_mid - 2.0 * bb_std

    # Volume: ratio degrades to NaN when the average is 0 (no inf).
    volume_sma = out["volume"].rolling(20).mean()
    out["volume_sma"] = volume_sma
    out["volume_ratio"] = out["volume"] / volume_sma.replace(0.0, np.nan)

    # Average True Range with Wilder smoothing.
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range"] = true_range
    out["atr"] = _rma(true_range, 14)

    # Stochastic: zero range degrades to NaN rather than exploding.
    low_14 = out["low"].rolling(14).min()
    high_14 = out["high"].rolling(14).max()
    price_range = (high_14 - low_14).replace(0.0, np.nan)
    out["stoch_k"] = 100.0 * ((close - low_14) / price_range)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    # Daily return: pct_change vs a zero previous close -> NaN, never inf.
    safe_prev = close.shift(1).replace(0.0, np.nan)
    out["daily_return"] = close / safe_prev - 1.0

    reported = [
        "rsi",
        "macd",
        "macd_signal",
        "sma_20",
        "sma_50",
        "sma_200",
        "bb_upper",
        "atr",
        "stoch_k",
    ]
    valid_starts: dict[str, str | None] = {}
    warmup_rows: dict[str, int] = {}
    for name in reported:
        col = out[name]
        first = col.first_valid_index()
        valid_starts[name] = (
            pd.Timestamp(first).tz_convert("UTC").date().isoformat()
            if first is not None
            else None
        )
        warmup_rows[name] = int(col.isna().cummin().sum())

    meta = {"valid_starts": valid_starts, "warmup_rows": warmup_rows}
    return out, meta
# ---------------------------------------------------------------------------
# ML features / label: shift(1) alignment, no look-ahead bias
# ---------------------------------------------------------------------------

def _build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that use only information known at the close of each bar.
    Callers must shift by one before using them to predict the next bar.
    """
    frame = df
    raw = pd.DataFrame(index=frame.index)
    raw["return_1d"] = frame["close"].pct_change(fill_method=None)
    raw["return_5d"] = frame["close"].pct_change(5, fill_method=None)

    for lag in (1, 2, 3, 5, 10):
        raw[f"close_lag_{lag}"] = frame["close"].shift(lag)
        raw[f"return_lag_{lag}"] = raw["return_1d"].shift(lag)
        raw[f"volume_lag_{lag}"] = frame["volume"].shift(lag)

    for window in (5, 10, 20, 50):
        raw[f"close_mean_{window}"] = frame["close"].rolling(window).mean()
        raw[f"close_std_{window}"] = frame["close"].rolling(window).std()
        raw[f"volume_mean_{window}"] = frame["volume"].rolling(window).mean()

    sma20 = frame["sma_20"].replace(0.0, np.nan)
    sma50 = frame["sma_50"].replace(0.0, np.nan)
    raw["price_vs_sma20"] = (frame["close"] - frame["sma_20"]) / sma20
    raw["price_vs_sma50"] = (frame["close"] - frame["sma_50"]) / sma50

    raw["volatility_10d"] = raw["return_1d"].rolling(10).std()
    raw["volatility_20d"] = raw["return_1d"].rolling(20).std()

    for name in ("rsi", "macd", "macd_signal", "macd_histogram", "atr",
                 "volume_ratio", "stoch_k", "stoch_d"):
        if name in frame.columns:
            raw[name] = frame[name]

    return raw.replace([np.inf, -np.inf], np.nan)


def build_ml_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """
    Build a leakage-free supervised dataset.

    Features describe information observable at the *close of bar t-1*
    (every feature column is shifted by one before alignment).  The label
    is the close-to-close return realized *during* bar t::

        label[t] = close[t] / close[t-1] - 1

    so a model can be used at the close of the latest bar to predict the
    next bar's return.  Feature frame and label share one index; rows are
    dropped with a single boolean mask so they can never be misaligned.

    Degenerate behaviour is explicit:
    * divisions by zero become NaN (denominators map 0 -> NaN);
    * +/-inf is replaced with NaN before rows are dropped;
    * feature columns that are entirely NaN are removed and reported;
    * if too few aligned rows remain the returned frame is empty.
    """
    # The label is the return realized over bar t (known only at close of t).
    label = df["close"].pct_change(fill_method=None).rename("target")

    # Features available at the *decision point* = close of bar t-1.
    features = _build_raw_features(df).shift(1)
    label = label.replace([np.inf, -np.inf], np.nan)

    # Drop feature columns that are NaN for every bar (e.g. sma_50 on a
    # short series) so no scaler/model ever sees a useless column.
    all_nan_cols = [c for c in features.columns if features[c].isna().all()]
    if all_nan_cols:
        features = features.drop(columns=all_nan_cols)
    feature_cols = list(features.columns)

    # Single mask over a joined frame => rows dropped identically on both sides.
    joined = features.join(label, how="inner")
    total_rows = len(joined)
    valid_mask = joined[feature_cols].notna().all(axis=1) & joined["target"].notna()
    aligned = joined[valid_mask]
    dropped_rows = int(total_rows - len(aligned))

    X = aligned[feature_cols].copy()
    y = aligned["target"].copy()

    meta = {
        "rows_before": int(total_rows),
        "rows_after": int(len(aligned)),
        "dropped_rows": dropped_rows,
        "all_nan_columns": all_nan_cols,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
    }
    return X, y, feature_cols, meta


@dataclass
class ModelResult:
    trained: bool
    reason: str | None = None
    model: Any = None
    scaler: Any = None
    feature_cols: list[str] = field(default_factory=list)
    train_score: float | None = None
    test_score: float | None = None
    predicted_return: float | None = None
    predicted_close: float | None = None
    samples: int = 0


def train_return_model(
    df: pd.DataFrame,
    min_rows: int = 60,
) -> tuple[ModelResult, dict[str, Any]]:
    """Train a RandomForest on next-bar returns using leakage-free rows."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    X, y, feature_cols, meta = build_ml_dataset(df)
    result = ModelResult(trained=False, feature_cols=feature_cols)

    if len(X) < min_rows:
        result.reason = f"only {len(X)} aligned rows (need {min_rows})"
        return result, meta

    # Chronological split: the most recent 20% acts as the test set.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) < 2:
        result.reason = "not enough rows for a test split"
        return result, meta

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.to_numpy())
    X_test_s = scaler.transform(X_test.to_numpy())

    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Prediction for the *next* bar uses the latest unshifted feature row:
    # information observable at the close of the current last bar.
    latest_row = _latest_feature_row(df, feature_cols)
    if latest_row is not None and np.isfinite(latest_row).all():
        predicted_return = float(
            model.predict(scaler.transform(latest_row.reshape(1, -1)))[0]
        )
        last_close = float(df["close"].iloc[-1])
        predicted_close = last_close * (1.0 + predicted_return)
    else:
        predicted_return = None
        predicted_close = None

    result = ModelResult(
        trained=True,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        train_score=float(model.score(X_train_s, y_train)),
        test_score=float(model.score(X_test_s, y_test)),
        predicted_return=predicted_return,
        predicted_close=predicted_close,
        samples=len(X),
    )
    return result, meta


def _latest_feature_row(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray | None:
    """Reconstruct the latest *unshifted* feature row for prediction."""
    raw = _build_raw_features(df)
    missing = [c for c in feature_cols if c not in raw.columns]
    if missing:
        return None
    row = raw[feature_cols].iloc[-1].to_numpy(dtype=float)
    row = np.where(np.isinf(row), np.nan, row)
    return row
# ---------------------------------------------------------------------------
# End-to-end pipeline with per-ticker failure isolation
# ---------------------------------------------------------------------------

@dataclass
class TickerResult:
    ticker: str
    ok: bool
    error: str | None = None
    cache_hit: bool = False
    stale_fallback: bool = False
    cache_key: str | None = None
    last_bar_date: str | None = None
    rows: int = 0
    dropped_rows: int = 0
    ml_rows: int = 0
    all_nan_columns: list[str] = field(default_factory=list)
    valid_starts: dict[str, str | None] = field(default_factory=dict)
    warmup_rows: dict[str, int] = field(default_factory=dict)
    trained: bool = False
    train_score: float | None = None
    test_score: float | None = None
    predicted_return: float | None = None
    predicted_close: float | None = None
    latest_close: float | None = None
    data: pd.DataFrame | None = None


def run_pipeline_for_ticker(
    ticker: str,
    interval: str = "1d",
    period: str = "2y",
    adjust: str = ADJUST_QFQ,
    cache: DataCache | None = None,
    fetcher: Callable[[str, str, str, str], pd.DataFrame] | None = None,
) -> TickerResult:
    """
    Fetch -> normalize -> indicators -> ML for one ticker.

    Any failure is captured on the returned :class:`TickerResult`; callers
    rendering a multi-ticker page can therefore keep going.
    """
    result = TickerResult(ticker=ticker.upper(), ok=False)
    download = fetcher or fetch_raw_ohlcv
    cache = cache or DataCache()

    try:
        ohlcv, cache_meta = cache.load_ohlcv(
            ticker,
            interval,
            adjust,
            lambda: download(ticker, interval, period, adjust),
        )
        result.cache_hit = cache_meta["cache_hit"]
        result.stale_fallback = cache_meta.get("stale_fallback", False)
        result.cache_key = cache_meta["cache_key"]
        result.last_bar_date = cache_meta["last_bar_date"]
        result.rows = cache_meta["rows"]

        enriched, ind_meta = calculate_indicators(ohlcv)
        result.valid_starts = ind_meta["valid_starts"]
        result.warmup_rows = ind_meta["warmup_rows"]

        model_result, ml_meta = train_return_model(enriched)
        result.dropped_rows = ml_meta["dropped_rows"]
        result.ml_rows = ml_meta["rows_after"]
        result.all_nan_columns = ml_meta["all_nan_columns"]
        result.trained = model_result.trained
        result.train_score = model_result.train_score
        result.test_score = model_result.test_score
        result.predicted_return = model_result.predicted_return
        result.predicted_close = model_result.predicted_close
        result.latest_close = float(enriched["close"].iloc[-1])
        result.data = enriched
        result.ok = True
    except Exception as exc:  # one bad ticker must never crash the page
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def run_pipeline(
    tickers: list[str],
    interval: str = "1d",
    period: str = "2y",
    adjust: str = ADJUST_QFQ,
    cache_dir: str | os.PathLike[str] = ".cache",
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    fetcher: Callable[[str, str, str, str], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Process many tickers independently and return a serializable summary."""
    cache = DataCache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
    results: list[TickerResult] = []
    for ticker in tickers:
        results.append(
            run_pipeline_for_ticker(
                ticker,
                interval=interval,
                period=period,
                adjust=adjust,
                cache=cache,
                fetcher=fetcher,
            )
        )

    summary = {
        "tickers": [t.upper() for t in tickers],
        "interval": interval,
        "period": period,
        "adjust": adjust,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "succeeded": [r.ticker for r in results if r.ok],
        "failed": [
            {"ticker": r.ticker, "error": r.error}
            for r in results
            if not r.ok
        ],
        "results": results,
    }
    return summary


def result_to_dict(result: TickerResult) -> dict[str, Any]:
    """JSON-safe projection of a single ticker result (no DataFrame)."""
    return {
        "ticker": result.ticker,
        "ok": result.ok,
        "error": result.error,
        "cache_hit": result.cache_hit,
        "stale_fallback": result.stale_fallback,
        "cache_key": result.cache_key,
        "last_bar_date": result.last_bar_date,
        "rows": result.rows,
        "dropped_rows": result.dropped_rows,
        "ml_rows": result.ml_rows,
        "all_nan_columns": result.all_nan_columns,
        "valid_starts": result.valid_starts,
        "warmup_rows": result.warmup_rows,
        "trained": result.trained,
        "train_score": result.train_score,
        "test_score": result.test_score,
        "predicted_return": result.predicted_return,
        "predicted_close": result.predicted_close,
        "latest_close": result.latest_close,
    }


def summary_to_json(summary: dict[str, Any]) -> str:
    payload = {
        k: v
        for k, v in summary.items()
        if k != "results"
    }
    payload["results"] = [result_to_dict(r) for r in summary["results"]]
    return json.dumps(payload, indent=2, default=str)
