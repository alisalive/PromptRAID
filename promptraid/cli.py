"""PromptRAID command-line interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from promptraid.db.store import ResultStore
from promptraid.harness.agent_runner import AgentRunner
from promptraid.harness.judge import RuleBasedJudge
from promptraid.harness.providers import get_provider
from promptraid.payloads.mutations import MutationCategory, MutationEngine
from promptraid.payloads.taxonomy import TAXONOMY, list_categories
from promptraid.reporting.export import export_html, export_json


@click.group()
def cli():
    """PromptRAID - automated prompt injection red-teaming for LLM agents."""


@cli.command("list-taxonomy")
def list_taxonomy():
    """Print all known payload categories and their ATLAS technique mapping."""
    for name in list_categories():
        tech = TAXONOMY[name]
        status = "verified" if tech.verified else "TODO"
        click.echo(f"{name:35s} {tech.technique_id:16s} {tech.name} [{status}]")


@cli.command("mutate")
@click.option("--payload", required=True, help="Base injection payload text.")
@click.option(
    "--category",
    "categories",
    multiple=True,
    type=click.Choice([c.value for c in MutationCategory]),
    help="Restrict mutation to specific categories (repeatable).",
)
@click.option("--output", type=click.Path(), help="Write JSON variants to this file.")
def mutate(payload: str, categories, output):
    """Generate mutated payload variants from a base injection payload."""
    engine = MutationEngine()
    selected = [MutationCategory(c) for c in categories] if categories else None
    variants = engine.generate(payload, categories=selected)
    data = [v.to_dict() for v in variants]

    if output:
        Path(output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        click.echo(f"Wrote {len(data)} variants to {output}")
    else:
        click.echo(json.dumps(data, indent=2))


@cli.command("judge")
@click.option("--transcript-file", required=True, type=click.Path(exists=True))
@click.option("--injected-directive", default=None)
def judge_cmd(transcript_file: str, injected_directive):
    """Classify a transcript file (plain text) as injection success/fail."""
    text = Path(transcript_file).read_text(encoding="utf-8")
    judge = RuleBasedJudge()
    result = judge.evaluate(text, injected_directive=injected_directive)
    click.echo(json.dumps(result.to_dict(), indent=2))


@cli.command("run")
@click.option("--payload", required=True, help="Payload to send to the target agent.")
@click.option(
    "--target",
    default="anthropic:claude-haiku-4-5",
    help="Target model as 'provider:model', e.g. groq:llama-3.3-70b-versatile, "
    "gemini:gemini-2.0-flash, openrouter:qwen/qwen-2.5-72b-instruct.",
)
@click.option("--system-prompt", default="You are a helpful assistant.")
def run_agent(payload: str, target: str, system_prompt: str):
    """Run a payload against a live target LLM agent through a tool-use loop."""
    provider = get_provider(target)
    runner = AgentRunner(provider=provider, system_prompt=system_prompt)
    transcript = runner.run(payload)
    click.echo(json.dumps(transcript.to_dict(), indent=2))


@cli.command("report")
@click.option("--run-id", required=True, type=int)
@click.option("--db-path", default=None, type=click.Path())
@click.option("--out-dir", default="reports", type=click.Path())
def report(run_id: int, db_path, out_dir):
    """Export a stored run's results to JSON and an HTML dark-theme report."""
    store = ResultStore(db_path=Path(db_path) if db_path else None)
    run = store.get_run(run_id)
    if not run:
        click.echo(f"No such run: {run_id}", err=True)
        sys.exit(1)

    results = store.get_results(run_id)
    out_dir_path = Path(out_dir)
    json_path = export_json(results, out_dir_path / f"run_{run_id}.json")
    html_path = export_html(
        results,
        out_dir_path / f"run_{run_id}.html",
        run_id=run_id,
        target_model=run.get("target_model", "unknown"),
    )
    click.echo(f"Exported JSON: {json_path}")
    click.echo(f"Exported HTML: {html_path}")
    store.close()


def main():
    cli()


if __name__ == "__main__":
    main()
