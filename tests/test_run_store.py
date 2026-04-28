# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the run_store persistence layer.

Covers:
  - DiagnoseStore.new_run()              — sequential ID, directory creation
  - DiagnoseStore.get_run()              — lookup by ID
  - DiagnoseStore.find_incomplete_run()  — most recent interrupted run
  - DiagnoseStore.list_runs()            — summary rows, newest-first order
  - DiagnoseRun.save_config/config()     — write and read back
  - DiagnoseRun.save_checkpoint/load_checkpoint() — atomic round-trip
  - DiagnoseRun.save_result/load_result()         — completion marker
  - DiagnoseRun.is_complete/is_interrupted/status() — status helpers
  - DiagnoseRun.summary_row()            — aggregated row
  - Directory naming convention          — "{id}_{timestamp}" format
  - Atomic write                         — tmp file renamed, not written in place
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aevyra_origin.run_store import CheckpointState, DiagnoseStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path) -> DiagnoseStore:
    return DiagnoseStore(root=tmp_path / ".origin")


def _sample_trace() -> dict:
    return {"nodes": [{"name": "plan", "input": "hi", "output": "ok"}]}


def _sample_checkpoint(run_id: str = "001") -> CheckpointState:
    return CheckpointState(
        run_id=run_id,
        rubric="Score 1 if correct.",
        method="all",
        score=0.4,
        trace_dict=_sample_trace(),
        completed_methods=["critic"],
        method_outputs={"critic": {"culprits": []}},
        llm_tokens=1200,
        ablation_calls=3,
    )


# ---------------------------------------------------------------------------
# DiagnoseStore.new_run()
# ---------------------------------------------------------------------------


class TestNewRun:
    def test_creates_directory(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        assert run.path.exists()
        assert run.path.is_dir()

    def test_run_id_starts_at_001(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        assert run.run_id == "001"

    def test_run_ids_are_sequential(self, tmp_store: DiagnoseStore) -> None:
        r1 = tmp_store.new_run()
        r2 = tmp_store.new_run()
        r3 = tmp_store.new_run()
        assert r1.run_id == "001"
        assert r2.run_id == "002"
        assert r3.run_id == "003"

    def test_directory_name_convention(self, tmp_store: DiagnoseStore) -> None:
        # Directories must be named "{id}_{timestamp}" e.g. "001_2026-04-24T10-32-15"
        run = tmp_store.new_run()
        name = run.path.name
        assert name.startswith("001_")
        # timestamp portion: YYYY-MM-DDTHH-MM-SS
        ts = name[4:]
        assert len(ts) == 19
        assert ts[4] == "-" and ts[7] == "-" and ts[10] == "T"

    def test_run_id_pads_to_three_digits(self, tmp_store: DiagnoseStore) -> None:
        for _ in range(10):
            tmp_store.new_run()
        run = tmp_store.new_run()
        assert run.run_id == "011"

    def test_creates_runs_dir_on_demand(self, tmp_store: DiagnoseStore) -> None:
        assert not tmp_store.runs_dir.exists()
        tmp_store.new_run()
        assert tmp_store.runs_dir.exists()


# ---------------------------------------------------------------------------
# DiagnoseStore.get_run()
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_returns_run_by_id(self, tmp_store: DiagnoseStore) -> None:
        created = tmp_store.new_run()
        found = tmp_store.get_run("001")
        assert found is not None
        assert found.run_id == created.run_id

    def test_returns_none_for_missing_id(self, tmp_store: DiagnoseStore) -> None:
        tmp_store.new_run()
        assert tmp_store.get_run("999") is None

    def test_returns_none_when_store_empty(self, tmp_store: DiagnoseStore) -> None:
        assert tmp_store.get_run("001") is None

    def test_finds_correct_run_among_multiple(self, tmp_store: DiagnoseStore) -> None:
        tmp_store.new_run()
        tmp_store.new_run()
        r3 = tmp_store.new_run()
        found = tmp_store.get_run("003")
        assert found is not None
        assert found.path == r3.path


# ---------------------------------------------------------------------------
# Config write/read
# ---------------------------------------------------------------------------


class TestConfig:
    def test_roundtrip(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_config(
            rubric="Be correct.",
            method="critic",
            score=0.5,
            trace_dict=_sample_trace(),
        )
        cfg = run.config()
        assert cfg is not None
        assert cfg["rubric"] == "Be correct."
        assert cfg["method"] == "critic"
        assert cfg["score"] == 0.5
        assert cfg["trace_dict"] == _sample_trace()
        assert cfg["run_id"] == "001"

    def test_config_returns_none_when_missing(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        assert run.config() is None

    def test_timestamp_present(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_config(rubric="r", method="all", score=0.0, trace_dict={})
        cfg = run.config()
        assert cfg is not None
        assert "timestamp" in cfg
        assert len(cfg["timestamp"]) > 0


# ---------------------------------------------------------------------------
# Checkpoint write/read (atomic round-trip)
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_roundtrip(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        state = _sample_checkpoint(run.run_id)
        run.save_checkpoint(state)

        loaded = run.load_checkpoint()
        assert loaded is not None
        assert loaded.run_id == state.run_id
        assert loaded.rubric == state.rubric
        assert loaded.method == state.method
        assert loaded.score == state.score
        assert loaded.trace_dict == state.trace_dict
        assert loaded.completed_methods == state.completed_methods
        assert loaded.method_outputs == state.method_outputs
        assert loaded.llm_tokens == state.llm_tokens
        assert loaded.ablation_calls == state.ablation_calls

    def test_load_returns_none_when_missing(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        assert run.load_checkpoint() is None

    def test_atomic_write_no_tmp_left_behind(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_checkpoint(_sample_checkpoint(run.run_id))
        # The .tmp file must not remain after a successful write
        assert not run.checkpoint_path.with_suffix(".tmp").exists()

    def test_checkpoint_is_valid_json(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_checkpoint(_sample_checkpoint(run.run_id))
        raw = run.checkpoint_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)  # must not raise
        assert "completed_methods" in parsed

    def test_timestamp_auto_filled(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        state = _sample_checkpoint(run.run_id)
        assert state.timestamp == ""
        run.save_checkpoint(state)
        loaded = run.load_checkpoint()
        assert loaded is not None
        assert loaded.timestamp != ""

    def test_checkpoint_overwrite(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        state = _sample_checkpoint(run.run_id)
        run.save_checkpoint(state)

        state.completed_methods = ["critic", "decomposition"]
        state.llm_tokens = 5000
        run.save_checkpoint(state)

        loaded = run.load_checkpoint()
        assert loaded is not None
        assert loaded.completed_methods == ["critic", "decomposition"]
        assert loaded.llm_tokens == 5000


# ---------------------------------------------------------------------------
# Result write/read
# ---------------------------------------------------------------------------


class TestResult:
    def test_roundtrip(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        data = {"summary": "all good", "culprits": []}
        run.save_result(data)
        loaded = run.load_result()
        assert loaded == data

    def test_load_returns_none_when_missing(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        assert run.load_result() is None

    def test_atomic_write_no_tmp_left_behind(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_result({"summary": "done"})
        assert not run.result_path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


class TestStatus:
    def test_running_when_no_files(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        assert run.status() == "running"
        assert not run.is_complete()
        assert not run.is_interrupted()

    def test_interrupted_when_checkpoint_only(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_checkpoint(_sample_checkpoint(run.run_id))
        assert run.status() == "interrupted"
        assert run.is_interrupted()
        assert not run.is_complete()

    def test_completed_when_result_exists(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_checkpoint(_sample_checkpoint(run.run_id))
        run.save_result({"summary": "done"})
        assert run.status() == "completed"
        assert run.is_complete()
        assert not run.is_interrupted()

    def test_completed_without_checkpoint(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_result({"summary": "done"})
        assert run.is_complete()
        assert not run.is_interrupted()


# ---------------------------------------------------------------------------
# DiagnoseStore.find_incomplete_run()
# ---------------------------------------------------------------------------


class TestFindIncompleteRun:
    def test_returns_none_when_empty(self, tmp_store: DiagnoseStore) -> None:
        assert tmp_store.find_incomplete_run() is None

    def test_returns_none_when_all_complete(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_result({"summary": "done"})
        assert tmp_store.find_incomplete_run() is None

    def test_finds_interrupted_run(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_checkpoint(_sample_checkpoint(run.run_id))
        found = tmp_store.find_incomplete_run()
        assert found is not None
        assert found.run_id == run.run_id

    def test_returns_most_recent_interrupted(self, tmp_store: DiagnoseStore) -> None:
        r1 = tmp_store.new_run()
        r1.save_checkpoint(_sample_checkpoint(r1.run_id))

        r2 = tmp_store.new_run()
        r2.save_checkpoint(_sample_checkpoint(r2.run_id))

        found = tmp_store.find_incomplete_run()
        assert found is not None
        assert found.run_id == "002"

    def test_skips_completed_runs(self, tmp_store: DiagnoseStore) -> None:
        r1 = tmp_store.new_run()
        r1.save_checkpoint(_sample_checkpoint(r1.run_id))
        r1.save_result({"summary": "done"})  # completed — must be skipped

        r2 = tmp_store.new_run()
        r2.save_checkpoint(_sample_checkpoint(r2.run_id))  # interrupted

        found = tmp_store.find_incomplete_run()
        assert found is not None
        assert found.run_id == "002"

    def test_returns_none_when_only_running(self, tmp_store: DiagnoseStore) -> None:
        # A run with neither checkpoint nor result is "running", not "interrupted"
        tmp_store.new_run()
        assert tmp_store.find_incomplete_run() is None


# ---------------------------------------------------------------------------
# DiagnoseStore.list_runs()
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_store(self, tmp_store: DiagnoseStore) -> None:
        assert tmp_store.list_runs() == []

    def test_returns_newest_first(self, tmp_store: DiagnoseStore) -> None:
        r1 = tmp_store.new_run()
        r1.save_config(rubric="r", method="all", score=0.1, trace_dict={})
        r2 = tmp_store.new_run()
        r2.save_config(rubric="r", method="all", score=0.2, trace_dict={})

        rows = tmp_store.list_runs()
        assert len(rows) == 2
        assert rows[0]["run_id"] == "002"
        assert rows[1]["run_id"] == "001"

    def test_row_contains_expected_fields(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_config(rubric="Be correct.", method="critic", score=0.5, trace_dict={})
        run.save_checkpoint(_sample_checkpoint(run.run_id))

        row = tmp_store.list_runs()[0]
        assert row["run_id"] == "001"
        assert row["status"] == "interrupted"
        assert row["method"] == "critic"
        assert row["score"] == 0.5
        assert row["llm_tokens"] == 1200
        assert row["llm_tokens_fmt"] == "1.2K"
        assert row["ablation_calls"] == 3
        assert "Be correct."[:60] in row["rubric_preview"]

    def test_status_reflected_correctly(self, tmp_store: DiagnoseStore) -> None:
        tmp_store.new_run()  # running
        r2 = tmp_store.new_run()
        r2.save_checkpoint(_sample_checkpoint(r2.run_id))  # interrupted
        r3 = tmp_store.new_run()
        r3.save_result({"summary": "done"})  # completed

        rows = {r["run_id"]: r["status"] for r in tmp_store.list_runs()}
        assert rows["001"] == "running"
        assert rows["002"] == "interrupted"
        assert rows["003"] == "completed"

    def test_llm_tokens_fmt_millions(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        state = _sample_checkpoint(run.run_id)
        state.llm_tokens = 1_500_000
        run.save_checkpoint(state)
        row = tmp_store.list_runs()[0]
        assert row["llm_tokens_fmt"] == "1.5M"


# ---------------------------------------------------------------------------
# summary_row()
# ---------------------------------------------------------------------------


class TestSummaryRow:
    def test_no_config_no_checkpoint(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        row = run.summary_row()
        assert row["run_id"] == "001"
        assert row["status"] == "running"
        assert row["method"] == "?"
        assert row["score"] is None
        assert row["llm_tokens"] == 0
        assert row["llm_tokens_fmt"] == "—"
        assert row["ablation_calls"] == 0

    def test_with_config_and_checkpoint(self, tmp_store: DiagnoseStore) -> None:
        run = tmp_store.new_run()
        run.save_config(rubric="Check it.", method="all", score=0.8, trace_dict={})
        state = _sample_checkpoint(run.run_id)
        run.save_checkpoint(state)

        row = run.summary_row()
        assert row["method"] == "all"
        assert row["score"] == 0.8
        assert row["llm_tokens"] == 1200
        assert row["ablation_calls"] == 3
        assert row["rubric_preview"] == "Check it."
