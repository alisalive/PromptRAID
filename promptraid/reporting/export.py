"""Exports PromptRAID results to JSON and a dark-theme HTML report."""
from __future__ import annotations

import json
import re
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


# --- Multi-provider benchmark report (summary table + example transcripts) ---
#
# This is a separate template/function from export_html above: export_html renders
# a flat list of individual mutation results (one row per payload variant), while
# this renders a live multi-provider benchmark run - a per-provider summary table
# plus full turn-by-turn example transcripts (user / assistant / tool_result),
# with the injected payload highlighted inline.

_BENCHMARK_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PromptRAID Benchmark Report</title>
<style>
  :root {{
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --muted: #8b949e;
    --accent: #ff2e63;
    --success: #3fb950;
    --partial: #d29922;
    --fail: #58a6ff;
    --inconclusive: #8b949e;
    --error: #f85149;
  }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', Consolas, monospace;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{ color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  h2 {{ color: var(--accent); margin-top: 2.5rem; }}
  h3 {{ margin-bottom: 0.5rem; }}
  .meta {{ color: var(--muted); margin-bottom: 1.5rem; }}
  .methodology {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.25rem;
    margin-bottom: 2rem;
    font-size: 0.9rem;
    line-height: 1.5;
  }}
  .methodology p {{ margin: 0.4rem 0; }}
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
  .badge-success {{ background: rgba(63, 185, 80, 0.15); color: var(--success); }}
  .badge-partial_compliance {{ background: rgba(210, 153, 34, 0.15); color: var(--partial); }}
  .badge-fail {{ background: rgba(88, 166, 255, 0.15); color: var(--fail); }}
  .badge-inconclusive {{ background: rgba(139, 148, 158, 0.15); color: var(--inconclusive); }}
  .badge-error {{ background: rgba(248, 81, 73, 0.15); color: var(--error); }}
  .transcript {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }}
  .turn {{
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.6rem;
    border-radius: 6px;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  .turn-label {{
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
    letter-spacing: 0.05em;
  }}
  .turn-user {{ background: rgba(88, 166, 255, 0.08); border-left: 3px solid #58a6ff; }}
  .turn-assistant {{ background: rgba(255, 46, 99, 0.08); border-left: 3px solid var(--accent); }}
  .turn-tool_result {{ background: rgba(210, 153, 34, 0.08); border-left: 3px solid #d29922; }}
  mark {{
    background: rgba(255, 46, 99, 0.35);
    color: #ffd7e0;
    padding: 0 0.15rem;
    border-radius: 3px;
  }}
  .judge-reason {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }}
  .signals {{ color: var(--muted); font-size: 0.8rem; }}
</style>
</head>
<body>
  <h1>PromptRAID Benchmark Report</h1>
  <div class="meta">Generated {generated_at}</div>

  <div class="methodology">
    <p><strong>Scenario: indirect tool-output injection.</strong> The target agent is given a
    <code>read_file</code> tool. When called, the tool returns file content that embeds an
    injected directive (<mark>{injected_payload}</mark>). The agent is asked, via a normal user
    message, to read the file and summarize it - the injection reaches the model indirectly
    through the tool result rather than the direct user message.</p>
    <p>Verdicts come from a rule-based judge that distinguishes <strong>neutral echo</strong>
    (the model factually quotes/summarizes the injected text without acting on it),
    <strong>adopted compliance</strong> (the model treats the injected text as a directive and
    complies or acknowledges following it), and <strong>refusal</strong> (the model recognizes and
    rejects the injection) - this distinction is what makes the verdicts below reliable, instead of
    flagging every verbatim quote of the payload as a "success".</p>
  </div>

  <h2>Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Provider</th><th>Model</th><th>Attempts</th><th>Success</th><th>Partial</th>
        <th>Fail</th><th>Inconclusive</th><th>Error</th>
      </tr>
    </thead>
    <tbody>
      {summary_rows}
    </tbody>
  </table>

  <h2>Example Transcripts</h2>
  {example_sections}
</body>
</html>
"""

_BENCHMARK_ROW_TEMPLATE = """<tr>
        <td>{provider}</td>
        <td>{model}</td>
        <td>{attempts}</td>
        <td>{success}</td>
        <td>{partial}</td>
        <td>{fail}</td>
        <td>{inconclusive}</td>
        <td>{error}</td>
      </tr>"""

_EXAMPLE_SECTION_TEMPLATE = """<div class="transcript">
    <h3>{target} <span class="badge badge-{verdict}">{verdict}</span></h3>
    {turns}
    <div class="judge-reason">{reason}</div>
    <div class="signals">Signals: {signals}</div>
  </div>"""


def _highlight(text: str, payload: str) -> str:
    """Escape `text` for HTML and wrap any occurrence of `payload` in <mark>."""
    escaped_text = _escape(text)
    escaped_payload = _escape(payload) if payload else ""
    if not escaped_payload:
        return escaped_text
    pattern = re.compile(re.escape(escaped_payload), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped_text)


def _render_turn(role_class: str, label: str, content: str, injected_payload: str) -> str:
    return (
        f'<div class="turn turn-{role_class}"><span class="turn-label">{_escape(label)}'
        f"</span>{_highlight(content, injected_payload)}</div>"
    )


def _render_messages(messages: List[Dict[str, Any]], injected_payload: str) -> str:
    """Render a RunTranscript's `messages` list as turn-by-turn HTML."""
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        if role == "user":
            parts.append(_render_turn("user", "User", str(m.get("content", "")), injected_payload))
        elif role == "assistant":
            text = m.get("text") or ""
            if text:
                parts.append(_render_turn("assistant", "Assistant", text, injected_payload))
            for tc in m.get("tool_calls") or []:
                call_desc = f"Calls {tc.get('name')}({json.dumps(tc.get('input'))})"
                parts.append(_render_turn("assistant", "Assistant tool call", call_desc, injected_payload))
        elif role == "tool_result":
            label = f"Tool result: {m.get('name', '')}"
            parts.append(
                _render_turn("tool_result", label, str(m.get("content", "")), injected_payload)
            )
    return "\n    ".join(parts)


def export_benchmark_html(
    summary_rows: List[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    injected_payload: str,
    output_path: Path,
) -> Path:
    """Render a dark-theme HTML benchmark report for a multi-provider live run and
    return the path.

    `summary_rows` items need keys: provider, model, attempts, success,
    partial_compliance, fail, inconclusive, error (as produced by
    scripts/live_demo.py's `summary_rows`).
    `examples` items need keys: target (str), transcript (RunTranscript.to_dict()),
    judge (JudgeResult.to_dict()) - one example per provider to render in full.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_html = "\n      ".join(
        _BENCHMARK_ROW_TEMPLATE.format(
            provider=_escape(r["provider"]),
            model=_escape(r["model"]),
            attempts=r["attempts"],
            success=r["success"],
            partial=r["partial_compliance"],
            fail=r["fail"],
            inconclusive=r["inconclusive"],
            error=r["error"],
        )
        for r in summary_rows
    )

    example_sections = "\n  ".join(
        _EXAMPLE_SECTION_TEMPLATE.format(
            target=_escape(ex["target"]),
            verdict=_escape(ex["judge"].get("verdict", "inconclusive")),
            turns=_render_messages(ex["transcript"].get("messages", []), injected_payload),
            reason=_escape(ex["judge"].get("reason", "")),
            signals=_escape(", ".join(ex["judge"].get("matched_signals", [])) or "none"),
        )
        for ex in examples
    )

    html = _BENCHMARK_HTML_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).isoformat(),
        injected_payload=_escape(injected_payload),
        summary_rows=summary_html or '<tr><td colspan="8">No results.</td></tr>',
        example_sections=example_sections or "<p>No example transcripts available.</p>",
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
