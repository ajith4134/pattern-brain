"""Phase 8 — slice 6: the LEADERBOARD (PLAN.md §12).

Persistent memory for the Stacked-DAG search: every ``(DAGSpec → score)`` trial is
recorded so the search (slice 7) and the LLM planner (slice 8) can read what has
been tried and what scored well. Stdlib ``sqlite3`` only (no MLflow), stored under
``data/leaderboard.sqlite`` (Rule 1, in-folder, gitignored).

Anti-overfit by construction (correction B, PLAN.md §12): every row stores both the
DAG's ``crps`` AND the persistence ``baseline_crps`` + ``crps_skill`` (skill > 0 means
it actually beat the naive baseline), plus an optional ``dsr`` (Deflated Sharpe,
which the search fills using the count of distinct specs tried). ``top(metric="dsr")``
ranks by the selection-bias-deflated score, never raw validation.

Domain-agnostic (Rule 23): stores generic scorecards; nothing crypto/candle here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, spec_sig TEXT, spec_json TEXT, dataset_id TEXT,
    n_nodes INTEGER, combiner TEXT, n_eval INTEGER,
    mean_crps REAL, baseline_crps REAL, crps_skill REAL,
    calib_ks REAL, calibrated INTEGER, dsr REAL, notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_skill ON runs(crps_skill);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id);
"""

_METRIC_ORDER = {"crps_skill": "DESC", "dsr": "DESC", "mean_crps": "ASC"}


class Leaderboard:
    """SQLite-backed record of DAG-search trials."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = os.path.join(_PROJECT_ROOT, "data", "leaderboard.sqlite")
        if db_path != ":memory:":
            db_path = os.path.abspath(db_path)
            if not (db_path == _PROJECT_ROOT or db_path.startswith(_PROJECT_ROOT + os.sep)):
                raise ValueError(f"Rule 1: leaderboard db must be inside {_PROJECT_ROOT}")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ write
    def record(self, spec, score: Dict[str, object], dataset_id: str = "default",
               dsr: Optional[float] = None, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (ts, spec_sig, spec_json, dataset_id, n_nodes, combiner, "
            "n_eval, mean_crps, baseline_crps, crps_skill, calib_ks, calibrated, dsr, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), spec.signature(), json.dumps(spec.to_dict()), dataset_id,
             int(spec.n_nodes), spec.combiner, int(score.get("n", 0)),
             _f(score.get("mean_crps")), _f(score.get("baseline_crps")),
             _f(score.get("crps_skill")), _f(score.get("calib_ks")),
             int(bool(score.get("calibrated", False))),
             (None if dsr is None else float(dsr)), notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------- read
    def top(self, k: int = 10, metric: str = "crps_skill",
            dataset_id: Optional[str] = None, dedupe: bool = True) -> List[Dict[str, object]]:
        if metric not in _METRIC_ORDER:
            raise ValueError(f"unknown metric {metric!r}; use {sorted(_METRIC_ORDER)}")
        where, params = "", []
        if dataset_id is not None:
            where = "WHERE dataset_id = ?"; params.append(dataset_id)
        if metric == "dsr":
            where = (where + (" AND" if where else "WHERE") + " dsr IS NOT NULL")
        rows = [dict(r) for r in self.conn.execute(
            f"SELECT * FROM runs {where} ORDER BY {metric} {_METRIC_ORDER[metric]}", params)]
        if dedupe:                                      # best row per distinct spec
            seen, out = set(), []
            for r in rows:
                if r["spec_sig"] in seen:
                    continue
                seen.add(r["spec_sig"]); out.append(r)
            rows = out
        return rows[:k]

    def best(self, metric: str = "crps_skill", dataset_id: Optional[str] = None) -> Optional[Dict]:
        t = self.top(1, metric=metric, dataset_id=dataset_id)
        return t[0] if t else None

    def get(self, run_id: int) -> Optional[Dict]:
        r = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(r) if r else None

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def summary(self, dataset_id: Optional[str] = None) -> Dict[str, object]:
        where, params = ("WHERE dataset_id = ?", [dataset_id]) if dataset_id else ("", [])
        n_runs = int(self.conn.execute(f"SELECT COUNT(*) FROM runs {where}", params).fetchone()[0])
        n_specs = int(self.conn.execute(
            f"SELECT COUNT(DISTINCT spec_sig) FROM runs {where}", params).fetchone()[0])
        beat = int(self.conn.execute(
            f"SELECT COUNT(*) FROM runs {where}{' AND' if where else 'WHERE'} crps_skill > 0",
            params).fetchone()[0])
        sig_dsr = int(self.conn.execute(
            f"SELECT COUNT(*) FROM runs {where}{' AND' if where else 'WHERE'} dsr >= 0.95",
            params).fetchone()[0])
        best = self.best(dataset_id=dataset_id)
        return {"n_runs": n_runs, "n_specs": n_specs, "n_beat_baseline": beat,
                "n_dsr_significant": sig_dsr,
                "best_skill": (best["crps_skill"] if best else None),
                "best_combiner": (best["combiner"] if best else None)}

    def close(self) -> None:
        self.conn.close()


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None                    # NaN → NULL
    except (TypeError, ValueError):
        return None


__all__ = ["Leaderboard"]
