"""
Flask entry point for the corrected stock pipeline.

* ``/dashboard`` - HTML page rendering every pipeline diagnostic plus the
  ML next-bar prediction for each ticker.
* ``/stats``     - the same payload as JSON (``?format=json`` or an
  ``Accept: application/json`` header), with a minimal HTML view otherwise.

Run with::

    python app.py

then open http://127.0.0.1:5000/dashboard
"""

from __future__ import annotations

import html
from typing import Any

from flask import Flask, jsonify, redirect, request

from stock_pipeline import (
    DEFAULT_CACHE_TTL_SECONDS,
    result_to_dict,
    run_pipeline,
)

app = Flask(__name__)

DEFAULT_TICKERS = "AAPL,MSFT,TSLA"


def _params() -> dict[str, str]:
    raw_tickers = request.args.get("tickers", DEFAULT_TICKERS)
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    return {
        "tickers": tickers or DEFAULT_TICKERS.split(","),
        "interval": request.args.get("interval", "1d"),
        "period": request.args.get("period", "2y"),
        "adjust": request.args.get("adjust", "qfq"),
    }


def _collect() -> dict[str, Any]:
    params = _params()
    summary = run_pipeline(
        tickers=params["tickers"],
        interval=params["interval"],
        period=params["period"],
        adjust=params["adjust"],
        ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
    )
    return summary


def _sparkline_svg(result: Any, width: int = 320, height: int = 70) -> str:
    """Inline SVG of the last 60 closes plus the predicted next close."""
    if result.data is None or "close" not in result.data.columns:
        return "<span class='muted'>no chart</span>"
    closes = result.data["close"].dropna().tail(60)
    if len(closes) < 5:
        return "<span class='muted'>insufficient points</span>"

    values = closes.to_numpy(dtype=float).tolist()
    if result.predicted_close is not None:
        values = values + [float(result.predicted_close)]

    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    pad = 6

    def xy(i: int, v: float) -> tuple[float, float]:
        x = pad + i * (width - 2 * pad) / max(len(values) - 1, 1)
        y = height - pad - (v - vmin) / span * (height - 2 * pad)
        return x, y

    pts = " ".join(f"{x:.1f},{y:.1f}" for i, v in enumerate(values)
                   for x, y in [xy(i, v)])
    last_x, last_y = xy(len(closes) - 1, closes.iloc[-1])
    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='spark'>",
        f"<polyline fill='none' stroke='#4ea1ff' stroke-width='1.5' points='{pts}'/>",
        f"<circle cx='{last_x:.1f}' cy='{last_y:.1f}' r='2.5' fill='#4ea1ff'/>",
    ]
    if result.predicted_close is not None:
        px, py = xy(len(values) - 1, float(result.predicted_close))
        parts.append(
            f"<line x1='{last_x:.1f}' y1='{last_y:.1f}' x2='{px:.1f}' "
            f"y2='{py:.1f}' stroke='#ff9f43' stroke-width='1.2' stroke-dasharray='3,2'/>"
        )
        parts.append(
            f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3' fill='#ff9f43'/>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f}%"


def _fmt_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _render_card(result: Any) -> str:
    if not result.ok:
        return (
            "<section class='card fail'>"
            f"<h2>{html.escape(result.ticker)}</h2>"
            "<p class='error'>fetch / processing failed - other tickers unaffected</p>"
            f"<pre>{html.escape(result.error or 'unknown error')}</pre>"
            "</section>"
        )

    indicator_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{result.warmup_rows.get(name, 0)}</td>"
        f"<td>{html.escape(str(result.valid_starts.get(name) or 'n/a'))}</td>"
        "</tr>"
        for name in (
            "rsi", "macd", "macd_signal", "sma_20", "sma_50",
            "sma_200", "bb_upper", "atr", "stoch_k",
        )
    )
    nan_cols = (
        ", ".join(html.escape(c) for c in result.all_nan_columns)
        if result.all_nan_columns else "-"
    )
    return f"""
    <section class="card ok">
      <div class="card-head">
        <h2>{html.escape(result.ticker)}</h2>
        <span class="badge {('stale' if result.stale_fallback else 'hit') if result.cache_hit else 'miss'}">
          {'STALE CACHE FALLBACK' if result.stale_fallback else ('cache HIT' if result.cache_hit else 'cache MISS')}
        </span>
      </div>
      {_sparkline_svg(result)}
      <div class="metrics">
        <div><span>last close</span><b>{result.latest_close:.2f}</b></div>
        <div><span>predicted next close</span><b>
          {f'{result.predicted_close:.2f}' if result.predicted_close is not None else 'n/a'}
        </b></div>
        <div><span>predicted return</span><b>{_fmt_pct(result.predicted_return)}</b></div>
        <div><span>train R2 / test R2</span><b>
          {_fmt_score(result.train_score)} / {_fmt_score(result.test_score)}
        </b></div>
      </div>
      <table class="kv">
        <tr><th>rows</th><td>{result.rows}</td></tr>
        <tr><th>last bar (UTC)</th><td>{html.escape(result.last_bar_date or 'n/a')}</td></tr>
        <tr><th>dropped (NaN / warm-up) rows</th><td>{result.dropped_rows}</td></tr>
        <tr><th>aligned ML rows</th><td>{result.ml_rows}</td></tr>
        <tr><th>all-NaN feature cols dropped</th><td>{nan_cols}</td></tr>
        <tr><th>cache key</th><td class="mono">{html.escape(result.cache_key or 'n/a')}</td></tr>
      </table>
      <details open>
        <summary>indicator warm-up &amp; valid start (UTC)</summary>
        <table class="grid">
          <tr><th>indicator</th><th>warm-up rows (head NaN)</th><th>first valid date</th></tr>
          {indicator_rows}
        </table>
      </details>
    </section>
    """


DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stock Pipeline Dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px;
         background: #0f1420; color: #e6ebf5; }}
  h1 {{ margin: 0 0 4px; }}
  .muted {{ color: #8a94a8; }}
  .grid-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
                 gap: 16px; margin-top: 18px; }}
  .card {{ background: #182033; border: 1px solid #2a3550;
          border-radius: 10px; padding: 16px; }}
  .card.fail {{ border-color: #7a2f3a; background: #2a1820; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; }}
  h2 {{ margin: 0; font-size: 20px; }}
  .badge {{ font-size: 12px; padding: 2px 8px; border-radius: 999px; }}
  .badge.hit {{ background: #143d2a; color: #4ade80; }}
  .badge.miss {{ background: #3d2f14; color: #fbbf24; }}
  .badge.stale {{ background: #4a3410; color: #fdba74; }}
  .error {{ color: #ff6b7a; }}
  .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 16px;
              margin: 12px 0; }}
  .metrics span {{ display: block; font-size: 11px; color: #8a94a8; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }}
  th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #2a3550;
           vertical-align: top; }}
  th {{ color: #8a94a8; font-weight: 600; }}
  .mono {{ font-family: Consolas, monospace; font-size: 11px; word-break: break-all; }}
  .spark {{ width: 100%; height: 70px; margin-top: 8px; background: #0f1420;
           border-radius: 6px; }}
  details {{ margin-top: 10px; }}
  summary {{ cursor: pointer; color: #8ab4ff; font-size: 13px; }}
  pre {{ white-space: pre-wrap; word-break: break-all; }}
  form {{ margin: 14px 0; }}
  input[type=text] {{ width: 320px; background: #0f1420; color: #e6ebf5;
                      border: 1px solid #2a3550; border-radius: 6px; padding: 6px 8px; }}
  button {{ background: #2563eb; color: white; border: 0; border-radius: 6px;
            padding: 7px 14px; cursor: pointer; }}
</style>
</head>
<body>
  <h1>Stock pipeline diagnostics</h1>
  <div class="muted">
    interval={interval} &middot; period={period} &middot; adjust={adjust}
    &middot; generated {generated_at} (UTC)
    &middot; <a href="/stats?{query}">/stats JSON</a>
  </div>

  <form method="get" action="/dashboard">
    <input type="text" name="tickers" value="{tickers_param}"
           placeholder="comma separated tickers">
    <input type="hidden" name="interval" value="{interval}">
    <input type="hidden" name="period" value="{period}">
    <input type="hidden" name="adjust" value="{adjust}">
    <button type="submit">Run pipeline</button>
  </form>

  {failed_block}
  <div class="grid-cards">{cards}</div>
</body>
</html>
"""


@app.get("/")
def index() -> Any:
    return redirect("/dashboard")


@app.get("/dashboard")
def dashboard() -> str:
    summary = _collect()
    cards = "".join(_render_card(r) for r in summary["results"])
    if summary["failed"]:
        items = "".join(
            "<li><b>{ticker}</b>: {error}</li>".format(
                ticker=html.escape(f["ticker"]),
                error=html.escape(f["error"] or "unknown error"),
            )
            for f in summary["failed"]
        )
        failed_block = (
            "<div class='card fail' style='margin-top:14px'>"
            "<b>Failed tickers (page keeps rendering):</b>"
            f"<ul>{items}</ul></div>"
        )
    else:
        failed_block = (
            "<p class='muted' style='margin-top:10px'>all tickers processed</p>"
        )

    params = _params()
    return DASHBOARD_TEMPLATE.format(
        interval=html.escape(str(summary["interval"])),
        period=html.escape(str(summary["period"])),
        adjust=html.escape(str(summary["adjust"])),
        generated_at=html.escape(str(summary["generated_at_utc"])),
        query=html.escape(request.query_string.decode()),
        tickers_param=html.escape(",".join(params["tickers"])),
        failed_block=failed_block,
        cards=cards,
    )


@app.get("/stats")
def stats() -> Any:
    summary = _collect()
    wants_json = (
        request.args.get("format") == "json"
        or request.accept_mimetypes.best == "application/json"
    )
    payload = {
        k: v for k, v in summary.items() if k != "results"
    }
    payload["results"] = [result_to_dict(r) for r in summary["results"]]
    if wants_json:
        return jsonify(payload)
    body = html.escape(
        __import__("json").dumps(payload, indent=2, default=str)
    )
    return (
        "<!doctype html><pre style='font-family:Consolas,monospace;"
        "padding:16px;white-space:pre-wrap'>" + body + "</pre>"
    )


if __name__ == "__main__":
    import os

    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
