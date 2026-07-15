"""Thin wrapper: `python scripts/live_demo.py` runs the same live demo as
`promptraid demo`. The actual logic lives in promptraid/demo.py so it's shared
between this script and the CLI.

Requires GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY / CEREBRAS_API_KEY
in the environment (loaded from .env).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (parent of this scripts/ dir) is importable so this
# script works regardless of the caller's cwd or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from promptraid.demo import run_demo


def main():
    run_demo()


if __name__ == "__main__":
    main()
