"""Pipeline data-quality tests: timezone, column names, warmup, alignment."""

import numpy as np
import pandas as pd
import pytest

from stock_pipeline import (
    QuoteCache,
    build_ml_dataset,
    compute_indicators,
    normalize_ohlcv,
)


def make_ohlcv(n=120, tz="US/Eastern", start="2024-01-02"):
    """Synthetic OHLCV frame mimicking raw yfinance output."""
    idx = pd.date_range(start, periods=n, freq="B", tz=tz)
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 3.0) + 0.3)
    return pd.DataFrame(
        {
            "Open": close - 0.4,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, n),
        },
        index=idx,
    )


def test_timezone_normalized_to_utc_sorted_deduped():
    raw = make_ohlcv()
    # Shuffle and duplicate a row to prove sorting + dedup happen.
    raw = pd.concat([raw, raw.iloc[[5]]])
    raw = raw.sample(frac=1.0, random_state=7)

    df, stats = normalize_ohlcv(raw)

    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert stats["duplicates_removed"] == 1
    assert stats["dropped_rows"] == 1
    # Conversion (not naive relabeling): 2024-01-02 00:00 US/Eastern -> 05:00 UTC.
    assert df.index[0] == pd.Timestamp("2024-01-02 05:00", tz="UTC")


def test_columns_lowercased_and_multiindex_flattened():
    raw = make_ohlcv(n=10)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])

    df, _ = normalize_ohlcv(raw)

    assert not isinstance(df.columns, pd.MultiIndex)
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)
    assert all(c == c.lower() for c in df.columns)


def test_indicator_warmup_rows_stay_nan():
    df, _ = normalize_ohlcv(make_ohlcv(n=120))
    out, stats = compute_indicators(df)

    # RSI (Wilder, period 14): first valid print at row 14, head never filled.
    assert out["rsi"].iloc[:14].isna().all()
    assert out["rsi"].iloc[14:].notna().all()
    assert stats["warmup_rows"]["rsi"] == 14

    # MACD: line needs 26 closes, signal needs 9 more prints -> 26+9-1 warmup.
    assert out["macd"].iloc[:25].isna().all()
    assert out["macd"].iloc[25:].notna().all()
    assert out["macd_signal"].iloc[:33].isna().all()
    assert out["macd_signal"].iloc[33:].notna().all()
    assert stats["warmup_rows"]["macd_signal"] == 33
    assert stats["valid_start"]["macd_signal"] == out.index[33]

    # Wilder smoothing check against a manual recursive computation.
    delta = df["close"].astype(float).diff()
    gain = delta.clip(lower=0.0).to_numpy()
    loss = (-delta.clip(upper=0.0)).to_numpy()
    avg_gain = gain[1:15].mean()
    avg_loss = loss[1:15].mean()
    for i in range(15, len(df)):
        avg_gain = (avg_gain * 13 + gain[i]) / 14
        avg_loss = (avg_loss * 13 + loss[i]) / 14
    expected = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    assert out["rsi"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_ml_features_shifted_and_labels_aligned():
    df, _ = normalize_ohlcv(make_ohlcv(n=120))
    out, _ = compute_indicators(df)

    X, y, next_features, stats = build_ml_dataset(out)

    # No lookahead: feature row at t equals the raw feature computed at t-1.
    raw_ret_1d = out["close"].astype(float).pct_change()
    common = X.index.intersection(raw_ret_1d.index)
    shifted = raw_ret_1d.shift(1).loc[common]
    pd.testing.assert_series_equal(X["ret_1d"].loc[common], shifted, check_names=False)

    # Features and labels share the exact same index after joint row drops.
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert stats["dropped_rows"] == len(out) - len(X)

    # Label is the next-day return of the close at the same timestamp.
    close = out["close"].astype(float)
    expected_y = close.pct_change().shift(-1).loc[X.index]
    pd.testing.assert_series_equal(y, expected_y, check_names=False)

    # Degenerate behavior: zero-volume column drops out, zero close -> row drop.
    broken = out.copy()
    broken["volume"] = 0.0
    broken.loc[broken.index[60], "close"] = 0.0
    X2, y2, _, stats2 = build_ml_dataset(broken)
    assert "volume_chg_1d" in stats2["dropped_columns"]
    assert "volume_chg_1d" not in X2.columns
    assert X2.index.equals(y2.index)
    assert not np.isinf(X2.to_numpy()).any()


def test_cache_key_tracks_last_bar_date():
    cache = QuoteCache(ttl=3600)
    df, _ = normalize_ohlcv(make_ohlcv(n=30, tz=None))

    key_v1 = cache.key_for("AAPL", "1d", True)
    assert key_v1 == ("AAPL", "1d", True, "no-bar")
    assert cache.get(key_v1) is None
    assert cache.misses == 1

    key_v2 = cache.put("AAPL", "1d", True, df)
    assert key_v2[-1] == df.index[-1].strftime("%Y-%m-%d")
    assert cache.get(cache.key_for("AAPL", "1d", True)) is not None
    assert cache.hits == 1

    # A new bar (incremental quote) changes the key -> old entry is missed.
    extended, _ = normalize_ohlcv(make_ohlcv(n=31, tz=None))
    cache.put("AAPL", "1d", True, extended)
    new_key = cache.key_for("AAPL", "1d", True)
    assert new_key != key_v2
    assert new_key[-1] == extended.index[-1].strftime("%Y-%m-%d")
    # Different adjustment mode or interval -> different key.
    assert cache.key_for("AAPL", "1d", False) != new_key
    assert cache.key_for("AAPL", "1h", True) != new_key
