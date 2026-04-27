#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        if value.endswith('Z'):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not input_dir.exists():
        return records

    for jsonl in sorted(input_dir.glob('*.jsonl')):
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def summarize(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {'count': 0, 'latest_ts': None})
    latest = datetime.min.replace(tzinfo=timezone.utc)

    for rec in records:
        model = str(rec.get('model_name', 'unknown'))
        backend = str(rec.get('backend', 'unknown'))
        ts = _parse_ts(rec.get('ts'))

        key = (model, backend)
        by_pair[key]['count'] += 1
        prev = by_pair[key]['latest_ts']
        if prev is None or ts > prev:
            by_pair[key]['latest_ts'] = ts

        if ts > latest:
            latest = ts

    rows = []
    for (model, backend), data in sorted(by_pair.items(), key=lambda x: (x[0][0], x[0][1])):
        ts = data['latest_ts']
        rows.append({
            'model_name': model,
            'backend': backend,
            'count': data['count'],
            'latest_ts': ts.isoformat().replace('+00:00', 'Z') if ts else '',
        })

    last_updated = latest.isoformat().replace('+00:00', 'Z') if records and latest != datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
    return rows, last_updated


def build_html(rows: list[dict[str, Any]], last_updated: str) -> str:
    safe_rows = [
        {
            'model_name': html.escape(str(r['model_name'])),
            'backend': html.escape(str(r['backend'])),
            'count': int(r['count']),
            'latest_ts': html.escape(str(r['latest_ts'])),
        }
        for r in rows
    ]

    table_rows = '\n'.join(
        f"<tr><td>{r['model_name']}</td><td>{r['backend']}</td><td>{r['count']}</td><td>{r['latest_ts']}</td></tr>"
        for r in safe_rows
    )
    if not table_rows:
        table_rows = '<tr><td colspan="4">No benchmark records yet.</td></tr>'

    chart_data = json.dumps([{'label': f"{r['model_name']} ({r['backend']})", 'count': r['count']} for r in safe_rows])

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>llama.cpp perf tracker</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f6f6f6; }}
    #chart {{ margin-top: 1.5rem; }}
    .bar {{ background: #4f46e5; color: white; margin: 0.25rem 0; padding: 0.25rem 0.5rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>llama.cpp perf tracker</h1>
  <p>Last updated: <strong>{html.escape(last_updated)}</strong></p>
  <div id="chart"><h2>Chart</h2></div>
  <table>
    <thead>
      <tr><th>Model</th><th>Backend</th><th>Runs</th><th>Latest timestamp</th></tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  <script>
    const data = {chart_data};
    const chart = document.getElementById('chart');
    if (data.length === 0) {{
      chart.insertAdjacentHTML('beforeend', '<p>No data yet.</p>');
    }} else {{
      const maxCount = Math.max(...data.map(d => d.count), 1);
      for (const d of data) {{
        const pct = Math.max(5, Math.round((d.count / maxCount) * 100));
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.width = `${{pct}}%`;
        bar.textContent = `${{d.label}} — ${{d.count}}`;
        chart.appendChild(bar);
      }}
    }}
  </script>
</body>
</html>
'''


def build_page(input_dir: Path, output_file: Path) -> None:
    records = load_records(input_dir)
    rows, last_updated = summarize(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(build_html(rows, last_updated))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build static perf tracker HTML from JSONL records')
    parser.add_argument('--input-dir', default='/mnt/nvme/perf-tracker/results', help='Directory containing monthly JSONL files')
    parser.add_argument('--output', default='/mnt/nvme/perf-tracker/site/index.html', help='Output HTML path')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_page(Path(args.input_dir), Path(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
