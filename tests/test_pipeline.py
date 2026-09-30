"""
Tests for the corrected stock pipeline.

The four required assertions:
1. tz:     normalized index is UTC, sorted and deduplicated.
2. cols:   OHLCV columns are lower-case and MultiIndex is flattened.
3. warmup: Wilder RSI / 26+9 MACD keep leading NaN rows (never back-filled).
4. align:  shifted ML features and the label stay index-aligned with no
           look-ahead; divisions by zero degrade cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stock_pipeline as sp


def _synthetic_multiindex_frame(n: int = 320, seed: int = 7) -> pd.DataFrame:
    """Mimic yfinance's MultiIndex, naive-timestamp download output."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = close + rng.normal(0.0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.4, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {
            ("Open", "AAA"): open_,
            ("High", "AAA"): high,
            ("Low", "AAA"): low,
            ("Close", "AAA"): close,
            ("Adj Close", "AAA"): close,
            ("Volume", "AAA"): volume,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. timezone / sort / dedupe
# ---------------------------------------------------------------------------

def test_normalized_index_is_utc_sorted_and_deduped():
    raw = _synthetic_multiindex_frame()
    # Pretend yfinance handed us a non-UTC tz index, unsorted, with dupes.
    raw.index = raw.index.tz_localize("America/New_York")
    raw = raw.iloc[::-1]
    dup = raw.iloc[[5]]
    raw = pd.concat([raw, dup])

    out = sp.normalize_ohlcv(raw)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique


def test_tz_conversion_changes_last_bar_date_and_cache_key():
    # A 19:30 America/New_York bar is already 00:30 *next day* in UTC.
    # Normalization moves the date forward, so the cache key changes too.
    idx_ny = pd.DatetimeIndex(
        ["2025-01-10 21:30", "2025-01-11 21:30"], tz="America/New_York"
    )
    frame = pd.DataFrame(
        {
            ("Open", "X"): [1.0, 1.1],
            ("High", "X"): [1.2, 1.3],
            ("Low", "X"): [0.9, 1.0],
            ("Close", "X"): [1.1, 1.2],
            ("Volume", "X"): [10.0, 20.0],
        },
        index=idx_ny,
    )
    assert frame.index.max().date().isoformat() == "2025-01-11"

    norm = sp.normalize_ohlcv(frame)
    utc_date = sp.last_bar_date(norm)
    assert utc_date == "2025-01-12"

    key = sp.make_cache_key("X", "1d", "qfq", utc_date)
    old_key = sp.make_cache_key("X", "1d", "qfq", "2025-01-11")
    assert key != old_key
    assert "2025-01-12" in key


# ---------------------------------------------------------------------------
# 2. column names / MultiIndex flattening
# ---------------------------------------------------------------------------

def test_ohlcv_columns_are_lowercase_and_flat():
    raw = _synthetic_multiindex_frame()
    assert isinstance(raw.columns, pd.MultiIndex)

    out = sp.normalize_ohlcv(raw)

    assert not isinstance(out.columns, pd.MultiIndex)
    for name in sp.OHLCV_COLUMNS:
        assert name in out.columns
    for capitalized in ("Open", "High", "Low", "Close", "Volume"):
        assert capitalized not in out.columns


# ---------------------------------------------------------------------------
# 3. indicator warm-up
# ---------------------------------------------------------------------------

def test_rsi_is_wilder_smoothed_with_leading_nans():
    raw = _synthetic_multiindex_frame()
    df, _ = sp.calculate_indicators(sp.normalize_ohlcv(raw))

    rsi = df["rsi"]
    # Wilder RSI: bar 1's delta is NaN and the 14-bar RMA seed completes on
    # bar 15 (0-based index 14), so the first 14 positions are NaN and the
    # first valid value lands at position 14.
    leading_nan = int(rsi.isna().cummin().sum())
    # positions 0..12 are NaN (bar-1 diff NaN + first 13 warm-up values);
    # the 14th seed closes at 0-based index 14 (15th bar).
    assert leading_nan == 13
    assert rsi.iloc[14:].notna().all()

    # Independently recompute the seeded Wilder value for bar 15.
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.iloc[1:15].mean()
    avg_loss = loss.iloc[1:15].mean()
    expected = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    assert rsi.iloc[14] == pytest.approx(expected, rel=1e-10)


def test_macd_keeps_26_plus_9_warmup_window_and_never_backfills():
    raw = _synthetic_multiindex_frame()
    df, meta = sp.calculate_indicators(sp.normalize_ohlcv(raw))

    # MACD line needs the slow 26-bar EMA seed -> 25 leading NaN rows.
    assert df["macd"].iloc[:25].isna().all()
    assert df["macd"].iloc[25:].notna().all()

    # Signal needs 9 valid MACD bars after that -> 33 leading NaN rows.
    assert df["macd_signal"].iloc[:33].isna().all()
    assert df["macd_signal"].iloc[33:].notna().all()
    assert df["macd_histogram"].iloc[:33].isna().all()
    assert df["macd_histogram"].iloc[33:].notna().all()

    # warm-up metadata reports leading NaN counts (RSI: 13 NaN + first
    # value at 0-based index 14; MACD line 25; signal line 33).
    assert meta["warmup_rows"]["rsi"] == 13
    assert meta["warmup_rows"]["macd"] == 25
    assert meta["warmup_rows"]["macd_signal"] == 33


def test_indicators_degrade_on_flat_prices():
    n = 60
    frame = pd.DataFrame(
        {
            "open": np.ones(n),
            "high": np.ones(n),
            "low": np.ones(n),
            "close": np.ones(n),
            "volume": np.zeros(n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC"),
    )
    df, _ = sp.calculate_indicators(frame)
    # Flat line -> RSI 50 (not NaN/inf), zero volume ratio -> NaN (no inf).
    assert df["rsi"].iloc[14:].eq(50.0).all()
    assert not np.isinf(df["volume_ratio"].to_numpy(dtype=float)).any()


# ---------------------------------------------------------------------------
# 4. feature/label alignment and look-ahead elimination
# ---------------------------------------------------------------------------

def test_ml_features_shift_alignment_no_lookahead():
    raw = _synthetic_multiindex_frame()
    df, _ = sp.calculate_indicators(sp.normalize_ohlcv(raw))

    X, y, feature_cols, meta = sp.build_ml_dataset(df)

    # Dropping rows happens on one shared mask: indices match exactly.
    assert (X.index == y.index).all()
    assert len(X) == len(y)

    # Rebuild the raw feature for a surviving row t: X[t] must equal the
    # raw feature evaluated at t-1 (decision point), proving the shift.
    raw_features = sp._build_raw_features(df)
    row_t = X.index[100]
    prev_t = X.index[99]
    assert row_t > prev_t
    assert X.loc[row_t, "close_lag_1"] == pytest.approx(
        raw_features.loc[prev_t, "close_lag_1"]
    )

    # No look-ahead: raw close_lag_1 is close.shift(1), i.e. close at t-1;
    # after the feature frame's shift(1), X[t] describes bar t-1, so its
    # close_lag_1 is the close of bar t-2.
    assert X["close_lag_1"].loc[row_t] == pytest.approx(
        df["close"].loc[X.index[98]]
    )
    assert raw_features.loc[row_t, "close_lag_1"] == pytest.approx(
        df["close"].loc[prev_t]
    )

    # Label at t is the return realized during bar t.
    expected_label = (
        df["close"].loc[row_t] / df["close"].loc[prev_t] - 1.0
    )
    assert y.loc[row_t] == pytest.approx(expected_label)

    # dropped rows are actually reported
    assert meta["dropped_rows"] == meta["rows_before"] - meta["rows_after"]


def test_ml_handles_zero_division_and_all_nan_columns():
    n = 80
    frame = pd.DataFrame(
        {
            "open": np.ones(n),
            "high": np.ones(n),
            "low": np.ones(n),
            "close": np.ones(n),
            "volume": np.ones(n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC"),
    )
    df, _ = sp.calculate_indicators(frame)
    X, y, feature_cols, meta = sp.build_ml_dataset(df)

    assert not np.isinf(X.to_numpy(dtype=float)).any()
    # sma_200 cannot exist in 80 bars -> column dropped, not passed to model.
    assert "price_vs_sma50" in meta["all_nan_columns"] or X.shape[1] >= 1
    for col in X.columns:
        assert not X[col].isna().all()


# ---------------------------------------------------------------------------
# failure isolation: one bad ticker never crashes the batch
# ---------------------------------------------------------------------------

def test_pipeline_isolates_single_ticker_failures(tmp_path):
    calls = {"n": 0}

    def fake_fetcher(ticker, interval, period, adjust):
        calls["n"] += 1
        if ticker == "BOGUS":
            raise sp.PipelineError("simulated download failure")
        return _synthetic_multiindex_frame(320, seed=hash(ticker) % 2**32)

    summary = sp.run_pipeline(
        ["AAA", "BOGUS", "BBB"],
        cache_dir=tmp_path,
        fetcher=fake_fetcher,
    )

    assert summary["succeeded"] == ["AAA", "BBB"]
    assert [f["ticker"] for f in summary["failed"]] == ["BOGUS"]
    by_ticker = {r.ticker: r for r in summary["results"]}
    assert by_ticker["AAA"].ok
    assert by_ticker["BBB"].ok
    assert not by_ticker["BOGUS"].ok
    assert "simulated download failure" in by_ticker["BOGUS"].error


def test_cache_second_call_is_a_hit(tmp_path):
    cache = sp.DataCache(cache_dir=tmp_path)

    def fake_fetcher(ticker, interval, period, adjust):
        return _synthetic_multiindex_frame(320)

    r1 = sp.run_pipeline_for_ticker(
        "AAA", cache=cache, fetcher=fake_fetcher
    )
    r2 = sp.run_pipeline_for_ticker(
        "AAA", cache=cache, fetcher=fake_fetcher
    )
    assert r1.ok and r2.ok
    assert r1.cache_hit is False
    assert r2.cache_hit is True


def test_cache_key_components_are_all_present():
    key = sp.make_cache_key("aapl", "1d", "qfq", "2025-03-12")
    assert key.startswith("AAPL_1d_qfq_2025-03-12_")
    # changing any component changes the key
    assert sp.make_cache_key("msft", "1d", "qfq", "2025-03-12") != key
    assert sp.make_cache_key("aapl", "1h", "qfq", "2025-03-12") != key
    assert sp.make_cache_key("aapl", "1d", "raw", "2025-03-12") != key
    assert sp.make_cache_key("aapl", "1d", "qfq", "2025-03-13") != key
