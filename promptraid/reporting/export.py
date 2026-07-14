"""Exports PromptRAID results to JSON and a dark-theme HTML report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PromptRAID Report</title>
<style>
  :root {{
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --muted: #8b949e;
    --accent: #ff2e63;
    --success: #f85149;
    --fail: #3fb950;
    --inconclusive: #d29922;
  }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', Consolas, monospace;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
  }}
  .meta {{ color: var(--muted); margin-bottom: 2rem; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--border);
  }}
  th, td {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
    vertical-align: top;
  }}
  th {{ color: var(--accent); text-transform: uppercase; font-size: 0.75rem; }}
  code {{
    background: #010409;
    color: #79c0ff;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    word-break: break-all;
  }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-success {{ background: rgba(248, 81, 73, 0.15); color: var(--success); }}
  .badge-fail {{ background: rgba(63, 185, 80, 0.15); color: var(--fail); }}
  .badge-inconclusive {{ background: rgba(210, 153, 34, 0.15); color: var(--inconclusive); }}
</style>
</head>
<body>
  <h1>PromptRAID Report</h1>
  <div class="meta">Generated {generated_at} &middot; Run #{run_id} &middot; Target: {target_model} &middot; {result_count} results</div>
  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th>ATLAS Technique</th>
        <th>Verdict</th>
        <th>Confidence</th>
        <th>Mutated Payload</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""

_ROW_TEMPLATE = """<tr>
        <td>{category}</td>
        <td>{technique_id}<br><span style="color:var(--muted)">{technique_name}</span></td>
        <td><span class="badge badge-{verdict}">{verdict}</span></td>
        <td>{confidence:.2f}</td>
        <td><code>{mutated_payload}</code></td>
      </tr>"""


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def export_json(results: List[Dict[str, Any]], output_path: Path) -> Path:
    """Write a list of result dicts to a JSON file and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path


def export_html(
    results: List[Dict[str, Any]],
    output_path: Path,
    run_id: Any = "-",
    target_model: str = "unknown",
) -> Path:
    """Render a dark-theme HTML report for a list of result dicts and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = "\n      ".join(
        _ROW_TEMPLATE.format(
            category=_escape(r.get("category", "")),
            technique_id=_escape(r.get("technique_id", "")),
            technique_name=_escape(r.get("technique_name", "")),
            verdict=_escape(r.get("verdict", "inconclusive")),
            confidence=float(r.get("confidence", 0.0)),
            mutated_payload=_escape(r.get("mutated_payload", "")),
        )
        for r in results
    )

    html = _HTML_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        target_model=_escape(target_model),
        result_count=len(results),
        rows=rows or '<tr><td colspan="5">No results.</td></tr>',
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
