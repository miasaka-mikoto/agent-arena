"""Persistence helpers for Agent Arena runs.

The benchmark is intentionally portable: JSON is the interchange format and
SQLite is a small local index for querying runs.  No network or user data is
ever touched by this module.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def jsonable(value: Any) -> Any:
    """Convert common Arena objects into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return str(value)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=indent, sort_keys=True),
        encoding="utf-8",
    )
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_sqlite(path: str | Path, runs: Iterable[Any]) -> Path:
    """Write a fresh local result index.

    The schema is deliberately denormalised enough to stay useful even when
    a future provider adds fields.  The full run and trace remain in JSON in
    the ``payload`` column.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and replace it only after a complete,
    # committed database has been closed.  A long tournament can contain
    # large public traces; writing directly to the final path used to leave a
    # schema-only or hot-journal file if the process was interrupted midway.
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    conn = sqlite3.connect(temp_path)
    completed = False
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS runs;
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                category TEXT,
                seed INTEGER,
                status TEXT,
                success INTEGER,
                steps INTEGER,
                invalid_actions INTEGER,
                retries INTEGER,
                latency_ms REAL,
                token_usage INTEGER,
                estimated_cost REAL,
                synthetic INTEGER,
                payload TEXT NOT NULL
            );
            CREATE INDEX idx_runs_agent ON runs(agent_id);
            CREATE INDEX idx_runs_task ON runs(task_id);
            CREATE INDEX idx_runs_category ON runs(category);
            """
        )
        for run in runs:
            item = jsonable(run)
            if not isinstance(item, Mapping):
                continue
            metrics = item.get("metrics") or {}
            task = item.get("task") or {}
            agent = item.get("agent") or {}
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, agent_id, task_id, category, seed, status, success,
                 steps, invalid_actions, retries, latency_ms, token_usage,
                 estimated_cost, synthetic, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(item.get("run_id", item.get("id", ""))),
                    str(item.get("agent_id", agent.get("id", "unknown"))),
                    str(item.get("task_id", task.get("task_id", task.get("id", "unknown")))),
                    str(item.get("category", task.get("category", "generic"))),
                    item.get("seed", task.get("seed")),
                    str(item.get("status", "unknown")),
                    int(bool(item.get("success", metrics.get("success", False)))),
                    int(item.get("steps", metrics.get("steps", 0)) or 0),
                    int(item.get("invalid_actions", metrics.get("invalid_actions", 0)) or 0),
                    int(item.get("retries", metrics.get("retries", 0)) or 0),
                    float(item.get("latency_ms", metrics.get("latency_ms", 0.0)) or 0.0),
                    int(item.get("token_usage", metrics.get("token_usage", 0)) or 0),
                    float(item.get("estimated_cost", metrics.get("estimated_cost", 0.0)) or 0.0),
                    int(bool(item.get("synthetic", agent.get("synthetic", True)))),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                ),
            )
        conn.commit()
        # Verify the on-disk structure before publishing it.  This is cheap
        # compared with serialising the traces and catches interrupted writes
        # during development/CI.
        check = conn.execute("PRAGMA integrity_check").fetchone()
        if not check or str(check[0]).lower() != "ok":
            raise sqlite3.DatabaseError(f"SQLite integrity check failed: {check!r}")
        completed = True
    finally:
        conn.close()
        if completed:
            os.replace(temp_path, target)
        else:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return target


def read_sqlite(path: str | Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT payload FROM runs ORDER BY rowid").fetchall()
        return [json.loads(row["payload"]) for row in rows]
    finally:
        conn.close()
