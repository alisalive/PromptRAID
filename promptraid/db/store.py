"""SQLite storage for PromptRAID runs and results."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path.home() / ".promptraid" / "promptraid.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target_model TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    category TEXT NOT NULL,
    technique_id TEXT NOT NULL,
    technique_name TEXT NOT NULL,
    base_payload TEXT NOT NULL,
    mutated_payload TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    transcript_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class ResultStore:
    """Thin SQLite wrapper for persisting PromptRAID runs and per-payload results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create_run(self, target_model: str, notes: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (created_at, target_model, notes) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), target_model, notes),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_result(
        self,
        run_id: int,
        category: str,
        technique_id: str,
        technique_name: str,
        base_payload: str,
        mutated_payload: str,
        verdict: str,
        confidence: float,
        transcript: Dict[str, Any],
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO results (
                run_id, category, technique_id, technique_name, base_payload,
                mutated_payload, verdict, confidence, transcript_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                category,
                technique_id,
                technique_name,
                base_payload,
                mutated_payload,
                verdict,
                confidence,
                json.dumps(transcript),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_results(self, run_id: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["transcript"] = json.loads(d.pop("transcript_json"))
            results.append(d)
        return results

    def list_runs(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]
