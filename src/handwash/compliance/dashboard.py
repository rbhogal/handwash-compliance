"""
Compliance Dashboard — lightweight Flask app.
Exposes a JSON API + a minimal HTML page showing aggregate compliance stats.

Run standalone:
    python -m src.handwash.compliance.dashboard

Or embed in the pipeline — call create_app() and run in a background thread.
"""

from __future__ import annotations
import json
import threading
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List

from flask import Flask, jsonify, render_template_string

# ── In-memory event store (replace with SQLite for persistence) ────────────────
_events: Deque[Dict] = deque(maxlen=500)
_lock = threading.Lock()


def record_event(
    track_id: int,
    compliant: bool,
    steps_detected: List[int],
    wash_duration_sec: float,
    message: str,
) -> None:
    """Thread-safe: append a wash event (called from the pipeline thread)."""
    with _lock:
        _events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "track_id": track_id,
            "compliant": compliant,
            "steps_detected": sorted(steps_detected),
            "wash_duration_sec": round(wash_duration_sec, 2),
            "message": message,
        })


def create_app() -> Flask:
    app = Flask(__name__)

    _HTML = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Handwash Compliance Dashboard</title>
      <meta http-equiv="refresh" content="5">
      <style>
        body { font-family: sans-serif; padding: 2rem; }
        .stat { font-size: 2rem; font-weight: bold; }
        .green { color: #16a34a; }
        .red   { color: #dc2626; }
        table  { border-collapse: collapse; width: 100%; margin-top: 1rem; }
        th, td { border: 1px solid #e5e7eb; padding: 0.5rem 1rem; text-align: left; }
        th     { background: #f3f4f6; }
      </style>
    </head>
    <body>
      <h1>Handwash Compliance</h1>
      <p>Total events: <span class="stat">{{ total }}</span></p>
      <p>Compliance rate:
        <span class="stat {{ 'green' if rate >= 80 else 'red' }}">{{ rate }}%</span>
      </p>
      <h2>Recent Events</h2>
      <table>
        <tr><th>Time</th><th>Track ID</th><th>Compliant</th><th>Duration (s)</th><th>Steps</th><th>Message</th></tr>
        {% for e in events %}
        <tr>
          <td>{{ e.timestamp }}</td>
          <td>{{ e.track_id }}</td>
          <td class="{{ 'green' if e.compliant else 'red' }}">{{ '✓' if e.compliant else '✗' }}</td>
          <td>{{ e.wash_duration_sec }}</td>
          <td>{{ e.steps_detected }}</td>
          <td>{{ e.message }}</td>
        </tr>
        {% endfor %}
      </table>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        with _lock:
            events = list(_events)
        total = len(events)
        compliant = sum(1 for e in events if e["compliant"])
        rate = round(100 * compliant / total, 1) if total else 0.0
        return render_template_string(
            _HTML, events=list(reversed(events))[:50], total=total, rate=rate
        )

    @app.route("/api/events")
    def api_events():
        with _lock:
            return jsonify(list(_events))

    @app.route("/api/stats")
    def api_stats():
        with _lock:
            events = list(_events)
        total = len(events)
        compliant = sum(1 for e in events if e["compliant"])
        return jsonify({
            "total": total,
            "compliant": compliant,
            "rate": round(100 * compliant / total, 1) if total else 0.0,
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
