"""Flask entry point for the AI stock dashboard.

Routes:
- /dashboard  rendered pipeline view (quotes, indicators, predictions, stats)
- /stats      machine-readable pipeline statistics (JSON)
"""

import os

from flask import Flask, jsonify, redirect, render_template_string, request

from stock_pipeline import DEFAULT_TICKERS, StockPipeline

app = Flask(__name__)
pipeline = StockPipeline(
    DEFAULT_TICKERS,
    period="1y",
    interval="1d",
    auto_adjust=True,
    synthetic_fallback=os.environ.get("STOCK_DASH_DEMO") == "1",
)

CSS = """
body { background:#0e1117; color:#e6e6e6; font-family:Segoe UI,Arial,sans-serif; margin:0; }
header { padding:18px 28px; background:#161b22; border-bottom:1px solid #30363d; }
header h1 { margin:0; font-size:22px; }
header p { margin:4px 0 0; color:#8b949e; font-size:13px; }
main { padding:20px 28px; }
.panel { background:#161b22; border:1px solid #30363d; border-radius:8px;
         padding:16px 20px; margin-bottom:20px; }
.panel h2 { margin:0 0 12px; font-size:16px; color:#58a6ff; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:6px 10px; border-bottom:1px solid #21262d; text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color:#8b949e; font-weight:600; }
.cards { display:flex; flex-wrap:wrap; gap:16px; }
.card { background:#161b22; border:1px solid #30363d; border-radius:8px;
        padding:14px 18px; min-width:340px; flex:1; }
.card h3 { margin:0 0 8px; font-size:17px; }
.metric { display:inline-block; margin:4px 14px 4px 0; }
.metric .label { color:#8b949e; font-size:11px; text-transform:uppercase; }
.metric .value { font-size:18px; font-weight:600; }
.up { color:#3fb950; } .down { color:#f85149; } .muted { color:#8b949e; }
.warn { background:#3d2e00; border:1px solid #9e6a03; border-radius:8px;
        padding:12px 18px; margin-bottom:20px; color:#e3b341; }
footer { padding:14px 28px; color:#8b949e; font-size:12px; }
"""

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI Stock Dashboard</title>
  <style>{{ css }}</style>
</head>
<body>
<header>
  <h1>&#128640; AI Stock Dashboard</h1>
  <p>UTC-normalized quotes &middot; Wilder RSI &middot; MACD 26+9 warmup &middot; shift(1) ML features</p>
</header>
<main>
  {% if stats.failed_tickers %}
  <div class="warn">
    &#9888;&#65039; Failed tickers (page keeps rendering): {{ stats.failed_tickers|join(', ') }}
  </div>
  {% endif %}
  {% if stats.synthetic_tickers %}
  <div class="warn">
    &#8505;&#65039; Live fetch unavailable; showing deterministic synthetic data for:
    {{ stats.synthetic_tickers|join(', ') }}
  </div>
  {% endif %}

  <div class="panel">
    <h2>Pipeline Stats</h2>
    <table>
      <tr><th>Ticker</th><th>Rows</th><th>Last Bar (UTC)</th><th>Dropped Rows</th>
          <th>RSI Warmup</th><th>MACD Warmup</th><th>Signal Warmup</th>
          <th>RSI Valid From</th><th>MACD Valid From</th><th>Signal Valid From</th>
          <th>ML Dropped</th><th>ML Samples</th></tr>
      {% for ticker, t in stats.tickers.items() %}
      <tr>
        <td><strong>{{ ticker }}</strong></td>
        <td>{{ t.rows }}</td>
        <td>{{ t.last_bar }}</td>
        <td>{{ t.dropped_rows }}</td>
        <td>{{ t.warmup_rows.rsi }}</td>
        <td>{{ t.warmup_rows.macd }}</td>
        <td>{{ t.warmup_rows.macd_signal }}</td>
        <td>{{ t.valid_start.rsi or 'n/a' }}</td>
        <td>{{ t.valid_start.macd or 'n/a' }}</td>
        <td>{{ t.valid_start.macd_signal or 'n/a' }}</td>
        <td>{{ t.ml_dropped_rows }}</td>
        <td>{{ t.ml_samples }}</td>
      </tr>
      {% endfor %}
    </table>
    <p class="muted">
      Cache hits: <strong>{{ stats.cache_hits }}</strong> &middot;
      misses: <strong>{{ stats.cache_misses }}</strong> &middot;
      failed tickers: <strong>{{ stats.failed_tickers|length }}</strong>
      {% if stats.failed_tickers %}({{ stats.failed_tickers|join(', ') }}){% endif %}
    </p>
  </div>

  <div class="cards">
    {% for ticker, card in cards.items() %}
    <div class="card">
      <h3>{{ ticker }}</h3>
      {{ card.sparkline|safe }}
      <div>
        <div class="metric"><div class="label">Last Close</div>
          <div class="value">${{ '%.2f'|format(card.last_close) }}</div></div>
        <div class="metric"><div class="label">RSI (Wilder)</div>
          <div class="value">{{ card.rsi }}</div></div>
        <div class="metric"><div class="label">MACD</div>
          <div class="value">{{ card.macd }}</div></div>
        <div class="metric"><div class="label">Signal</div>
          <div class="value">{{ card.macd_signal }}</div></div>
      </div>
      <div>
        {% if card.prediction %}
        <div class="metric"><div class="label">Next-Day Prediction</div>
          <div class="value {{ 'up' if card.prediction.predicted_return >= 0 else 'down' }}">
            ${{ '%.2f'|format(card.prediction.predicted_close) }}
            ({{ '%+.2f%%'|format(card.prediction.predicted_return * 100) }})</div></div>
        <div class="metric"><div class="label">Test R&sup2;</div>
          <div class="value">{{ card.test_score }}</div></div>
        {% else %}
        <div class="metric"><div class="label">Next-Day Prediction</div>
          <div class="value muted">insufficient data</div></div>
        {% endif %}
      </div>
      <table>
        <tr><th>Date (UTC)</th><th>Close</th><th>RSI</th><th>MACD</th><th>Signal</th></tr>
        {% for row in card.tail %}
        <tr>
          <td>{{ row.date }}</td>
          <td>{{ row.close }}</td>
          <td>{{ row.rsi }}</td>
          <td>{{ row.macd }}</td>
          <td>{{ row.macd_signal }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
    {% endfor %}
  </div>
</main>
<footer>Educational use only. Not financial advice.</footer>
</body>
</html>
"""


def _sparkline(series, width=320, height=60):
    """Render a tiny inline SVG sparkline for a price series."""
    values = [float(v) for v in series.dropna()]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = width / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - (v - lo) / span * (height - 4) - 2:.1f}"
        for i, v in enumerate(values)
    )
    color = "#3fb950" if values[-1] >= values[0] else "#f85149"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>'
        f"</svg>"
    )


def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _build_cards(results):
    cards = {}
    for ticker, result in results.items():
        data = result["data"]
        tail = data.tail(8)
        prediction = result["prediction"]
        if prediction and prediction.get("test_score") is not None:
            prediction = dict(prediction, test_score=f"{prediction['test_score']:.3f}")
        cards[ticker] = {
            "last_close": float(data["close"].iloc[-1]),
            "rsi": _fmt(data["rsi"].iloc[-1], 1),
            "macd": _fmt(data["macd"].iloc[-1], 3),
            "macd_signal": _fmt(data["macd_signal"].iloc[-1], 3),
            "sparkline": _sparkline(data["close"].tail(60)),
            "prediction": prediction,
            "tail": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": f"{row['close']:.2f}",
                    "rsi": _fmt(row["rsi"], 1),
                    "macd": _fmt(row["macd"], 3),
                    "macd_signal": _fmt(row["macd_signal"], 3),
                }
                for idx, row in tail.iterrows()
            ],
        }
    return cards


@app.route("/")
def index():
    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    tickers = request.args.get("tickers")
    if tickers:
        pipeline.tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if request.args.get("demo") == "1":
        pipeline.synthetic_fallback = True
    results = pipeline.run()
    return render_template_string(
        DASHBOARD_HTML,
        css=CSS,
        stats=pipeline.stats,
        cards=_build_cards(results),
    )


@app.route("/stats")
def stats():
    if not pipeline.stats["tickers"] and not pipeline.stats["failed_tickers"]:
        pipeline.run()
    return jsonify(pipeline.stats)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
