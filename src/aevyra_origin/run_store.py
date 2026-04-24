# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistence layer for Origin diagnostic runs.

Run history is stored under a configurable root directory (default ``.origin/``):

    .origin/
      diagnoses/
        001_2026-04-24T10-32-15/
          config.json       — rubric, method, score, serialized trace, timestamp
          checkpoint.json   — per-method results as they complete (written atomically)
          result.json       — final Attribution dict (written on completion)
        002_2026-04-24T11-05-00/
          ...

A run that has a ``checkpoint.json`` but no ``result.json`` was interrupted and
can be resumed. :class:`DiagnoseStore` finds the latest such run via
:meth:`find_incomplete_run`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic JSON write — write to a temp file then rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.rename(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_tokens(n: int) -> str:
    """Human-readable token count: 1234 → '1.2K', 1234567 → '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------


@dataclass
class CheckpointState:
    """Intermediate state written after each attribution method completes.

    The checkpoint is the resume anchor: if Origin is interrupted mid-run
    (e.g. during a long ablation sweep), restarting with the same
    ``DiagnoseRun`` will load this checkpoint and skip whichever methods
    already completed.

    Fields:
        run_id:            Sequential run identifier (e.g. ``"001"``).
        rubric:            Evaluation rubric passed to the attribution methods.
        method:            Requested attribution method (``"all"``, etc.).
        score:             Judge score being explained.
        trace_dict:        Serialized ``AgentTrace`` (``AgentTrace.to_dict()``).
        completed_methods: List of method names that have finished.
        method_outputs:    Raw output dict per completed method (culprits as
                           plain dicts, not ``NodeAttribution`` objects).
        llm_tokens:        Cumulative LLM tokens consumed so far.
        ablation_calls:    Number of runner+judge invocations in ablation.
        timestamp:         ISO timestamp of the last checkpoint write.
    """

    run_id: str
    rubric: str
    method: str
    score: float
    trace_dict: dict[str, Any]
    completed_methods: list[str] = field(default_factory=list)
    method_outputs: dict[str, Any] = field(default_factory=dict)
    llm_tokens: int = 0
    ablation_calls: int = 0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# DiagnoseRun — handle for one run's directory
# ---------------------------------------------------------------------------


class DiagnoseRun:
    """Handle for one diagnostic run's on-disk directory.

    Created by :class:`DiagnoseStore` — don't instantiate directly.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id: str = path.name.split("_")[0]
        self.config_path = path / "config.json"
        self.checkpoint_path = path / "checkpoint.json"
        self.result_path = path / "result.json"

    # ------------------------------------------------------------------
    # Config (written once at run start)
    # ------------------------------------------------------------------

    def save_config(
        self,
        *,
        rubric: str,
        method: str,
        score: float,
        trace_dict: dict[str, Any],
    ) -> None:
        """Write the run config. Called once when a new run starts."""
        self.path.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.config_path,
            {
                "run_id": self.run_id,
                "rubric": rubric,
                "method": method,
                "score": score,
                "trace_dict": trace_dict,
                "timestamp": _now_iso(),
            },
        )

    def config(self) -> dict[str, Any] | None:
        if not self.config_path.exists():
            return None
        return _read_json(self.config_path)

    # ------------------------------------------------------------------
    # Checkpoint (written after each method completes)
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: CheckpointState) -> None:
        """Atomically write the current checkpoint state."""
        if not state.timestamp:
            state.timestamp = _now_iso()
        _write_json(
            self.checkpoint_path,
            {
                "run_id": state.run_id,
                "rubric": state.rubric,
                "method": state.method,
                "score": state.score,
                "trace_dict": state.trace_dict,
                "completed_methods": state.completed_methods,
                "method_outputs": state.method_outputs,
                "llm_tokens": state.llm_tokens,
                "ablation_calls": state.ablation_calls,
                "timestamp": state.timestamp,
            },
        )

    def load_checkpoint(self) -> CheckpointState | None:
        """Load the checkpoint, or ``None`` if none exists yet."""
        if not self.checkpoint_path.exists():
            return None
        d = _read_json(self.checkpoint_path)
        return CheckpointState(
            run_id=d["run_id"],
            rubric=d["rubric"],
            method=d["method"],
            score=float(d["score"]),
            trace_dict=d["trace_dict"],
            completed_methods=list(d.get("completed_methods", [])),
            method_outputs=dict(d.get("method_outputs", {})),
            llm_tokens=int(d.get("llm_tokens", 0)),
            ablation_calls=int(d.get("ablation_calls", 0)),
            timestamp=d.get("timestamp", ""),
        )

    # ------------------------------------------------------------------
    # Result (written on completion)
    # ------------------------------------------------------------------

    def save_result(self, result_dict: dict[str, Any]) -> None:
        """Write the final Attribution dict. Marks the run as complete."""
        _write_json(self.result_path, result_dict)

    def load_result(self) -> dict[str, Any] | None:
        if not self.result_path.exists():
            return None
        return _read_json(self.result_path)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        return self.result_path.exists()

    def is_interrupted(self) -> bool:
        return self.checkpoint_path.exists() and not self.result_path.exists()

    def status(self) -> str:
        if self.result_path.exists():
            return "completed"
        if self.checkpoint_path.exists():
            return "interrupted"
        return "running"

    def summary_row(self) -> dict[str, Any]:
        """One-row summary dict for :meth:`DiagnoseStore.list_runs`."""
        cfg = self.config() or {}
        ckpt = self.load_checkpoint()
        return {
            "run_id": self.run_id,
            "status": self.status(),
            "method": cfg.get("method", "?"),
            "score": cfg.get("score"),
            "completed_methods": ckpt.completed_methods if ckpt else [],
            "llm_tokens": ckpt.llm_tokens if ckpt else 0,
            "llm_tokens_fmt": _fmt_tokens(ckpt.llm_tokens) if ckpt else "—",
            "ablation_calls": ckpt.ablation_calls if ckpt else 0,
            "rubric_preview": (cfg.get("rubric") or "")[:60],
            "timestamp": cfg.get("timestamp", ""),
        }


# ---------------------------------------------------------------------------
# DiagnoseStore — manages a directory of runs
# ---------------------------------------------------------------------------


class DiagnoseStore:
    """Manages a directory of Origin diagnostic run records.

    Args:
        root: Root directory for run storage. Defaults to ``.origin`` in
              the current working directory. Created on first use.

    Usage::

        store = DiagnoseStore()                     # uses .origin/
        run   = store.new_run()                     # creates 001_.../ dir
        # ... pass run to Origin.diagnose() ...
        store.find_incomplete_run()                 # resume latest interrupted run
        for row in store.list_runs(): print(row)    # audit trail
    """

    def __init__(self, root: str | Path = ".origin") -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "diagnoses"

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def new_run(self) -> DiagnoseRun:
        """Create a new run directory with the next sequential ID."""
        run_id = self._next_run_id()
        timestamp = _now_iso()
        path = self.runs_dir / f"{run_id}_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return DiagnoseRun(path)

    def get_run(self, run_id: str) -> DiagnoseRun | None:
        """Look up a run by ID (e.g. ``"001"``). Returns ``None`` if not found."""
        for d in self._run_dirs():
            if d.name.startswith(f"{run_id}_") or d.name == run_id:
                return DiagnoseRun(d)
        return None

    def find_incomplete_run(self) -> DiagnoseRun | None:
        """Return the most recent interrupted run (has checkpoint, no result).

        Used to implement ``--resume``: call this, then pass the returned
        ``DiagnoseRun`` to :meth:`Origin.diagnose` as ``run=``.
        """
        for d in reversed(self._run_dirs()):
            run = DiagnoseRun(d)
            if run.is_interrupted():
                return run
        return None

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary row for every run, newest first."""
        return [DiagnoseRun(d).summary_row() for d in reversed(self._run_dirs())]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_dirs(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            [d for d in self.runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    def _next_run_id(self) -> str:
        ids = []
        for d in self._run_dirs():
            m = re.match(r"^(\d+)_", d.name)
            if m:
                ids.append(int(m.group(1)))
        return f"{(max(ids, default=0) + 1):03d}"


__all__ = [
    "CheckpointState",
    "DiagnoseRun",
    "DiagnoseStore",
]
