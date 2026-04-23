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

"""Tests for ablation.py — causal ablation attribution method."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from aevyra_witness import AgentTrace, TraceNode
from aevyra_witness.trace import KIND_REASON, KIND_TOOL

from aevyra_origin.ablation import (
    AblationError,
    _build_placeholder,  # type: ignore[attr-defined]
    run_ablation,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def linear_trace() -> AgentTrace:
    return AgentTrace(nodes=[
        TraceNode("classify", id="a", input="ticket", output="billing"),
        TraceNode("retrieve", id="r", input="billing", output="docs"),
        TraceNode("answer", id="c", input="ticket+docs", output="reply", optimize=True),
    ])


def dag_trace() -> AgentTrace:
    return AgentTrace(nodes=[
        TraceNode("plan", id="p1", kind=KIND_REASON, prompt_id="planner",
                  step=1, input="q", output="call tools", optimize=True),
        TraceNode("search", id="t1", kind=KIND_TOOL, parent_id="p1",
                  input={"q": "x"}, output="result"),
        TraceNode("plan", id="p2", kind=KIND_REASON, prompt_id="planner",
                  step=2, input="ctx", output="respond", optimize=True),
    ])


def stub_runner(trace: AgentTrace, overrides: dict[str, Any]) -> AgentTrace:
    """Apply overrides and return a new AgentTrace with forced outputs."""
    new_nodes = []
    for n in trace.nodes:
        if n.id in overrides:
            new_nodes.append(TraceNode(
                name=n.name, id=n.id, input=n.input,
                output=overrides[n.id], kind=n.kind,
                prompt_id=n.prompt_id, parent_id=n.parent_id,
            ))
        else:
            new_nodes.append(n)
    return AgentTrace(nodes=new_nodes, ideal=trace.ideal)


def judge_from_deltas(deltas: dict[str, float], baseline: float = 0.9):
    """Build a judge where ablating span id X drops the score by deltas[X]."""
    def judge(t: AgentTrace) -> float:
        s = baseline
        for n in t.nodes:
            if n.id in deltas and not n.output:
                s -= deltas[n.id]
        return max(0.0, s)
    return judge


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunAblationHappyPath:
    def test_culprits_ordered_by_absolute_delta(self):
        trace = linear_trace()
        # a has biggest impact, c next, r smallest
        deltas = {"a": 0.5, "c": 0.3, "r": 0.2}
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas, baseline=0.9),
            score_range=(0.0, 1.0),
        )
        culprits = result["culprits"]
        assert len(culprits) == 3
        # Sorted by |delta| descending
        assert culprits[0].node_id == "a"
        assert culprits[1].node_id == "c"
        assert culprits[2].node_id == "r"

    def test_confidence_equals_normalized_delta(self):
        trace = linear_trace()
        deltas = {"a": 0.5}  # only "a" has impact
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas, baseline=0.9),
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        a_culprit = next((c for c in result["culprits"] if c.node_id == "a"), None)
        assert a_culprit is not None
        # normalized_delta = 0.5 / 1.0 = 0.5
        assert abs(a_culprit.confidence - 0.5) < 1e-9

    def test_severity_thresholds_primary(self):
        trace = linear_trace()
        # delta=0.5 → normalized=0.5 → primary
        deltas = {"a": 0.5}
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas, baseline=0.9),
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        a_culprit = next(c for c in result["culprits"] if c.node_id == "a")
        assert a_culprit.severity == "primary"

    def test_severity_thresholds_contributing(self):
        trace = linear_trace()
        # delta=0.3 → normalized=0.3 → contributing
        deltas = {"c": 0.3}
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas, baseline=0.9),
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        c_culprit = next(c for c in result["culprits"] if c.node_id == "c")
        assert c_culprit.severity == "contributing"

    def test_severity_thresholds_minor(self):
        trace = linear_trace()
        # delta=0.1 → normalized=0.1 → minor
        deltas = {"r": 0.1}
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas, baseline=0.9),
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        r_culprit = next(c for c in result["culprits"] if c.node_id == "r")
        assert r_culprit.severity == "minor"

    def test_prompt_id_carried_through(self):
        trace = dag_trace()
        deltas = {"p1": 0.5}
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas, baseline=0.9),
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        p1_culprit = next((c for c in result["culprits"] if c.node_id == "p1"), None)
        assert p1_culprit is not None
        assert p1_culprit.prompt_id == "planner"


# ---------------------------------------------------------------------------
# Harmful spans
# ---------------------------------------------------------------------------


class TestHarmfulSpans:
    def test_harmful_span_appears_as_culprit(self):
        """A span whose removal *improves* the score is harmful."""
        trace = linear_trace()
        # "a" is harmful: ablating it improves score from 0.5 to 0.9
        def bad_judge(t: AgentTrace) -> float:
            for n in t.nodes:
                if n.id == "a" and not n.output:
                    return 0.9  # improved when "a" is gone
            return 0.5  # baseline is bad because "a" is present

        result = run_ablation(
            trace=trace,
            score=0.5,
            rubric="Quality.",
            runner=stub_runner,
            judge=bad_judge,
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        a_culprit = next((c for c in result["culprits"] if c.node_id == "a"), None)
        assert a_culprit is not None
        assert "actively degrading" in a_culprit.reasoning or "harmful" in a_culprit.reasoning

    def test_harmful_span_has_absolute_confidence(self):
        """Confidence should be based on |raw_delta|, not raw_delta."""
        trace = linear_trace()
        def harmful_judge(t: AgentTrace) -> float:
            for n in t.nodes:
                if n.id == "a" and not n.output:
                    return 0.9  # improved
            return 0.5

        result = run_ablation(
            trace=trace,
            score=0.5,
            rubric="Quality.",
            runner=stub_runner,
            judge=harmful_judge,
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        a_culprit = next(c for c in result["culprits"] if c.node_id == "a")
        # |delta| = |0.5 - 0.9| = 0.4, normalized = 0.4
        assert a_culprit.confidence > 0


# ---------------------------------------------------------------------------
# _build_placeholder
# ---------------------------------------------------------------------------


class TestBuildPlaceholder:
    def test_string_output_returns_empty_string(self):
        node = TraceNode("n", input="x", output="hello")
        trace = AgentTrace(nodes=[node])
        result = _build_placeholder(node, "null", trace)
        assert result == ""

    def test_list_output_returns_empty_list(self):
        node = TraceNode("n", input="x", output=[1, 2, 3])
        trace = AgentTrace(nodes=[node])
        result = _build_placeholder(node, "null", trace)
        assert result == []

    def test_dict_output_returns_empty_dict(self):
        node = TraceNode("n", input="x", output={"key": "val"})
        trace = AgentTrace(nodes=[node])
        result = _build_placeholder(node, "null", trace)
        assert result == {}

    def test_none_output_returns_none(self):
        node = TraceNode("n", input="x", output=None)
        trace = AgentTrace(nodes=[node])
        result = _build_placeholder(node, "null", trace)
        assert result is None

    def test_ideal_placeholder_uses_trace_ideal(self):
        node = TraceNode("n", input="x", output="foo")
        trace = AgentTrace(nodes=[node], ideal="ideal output")
        result = _build_placeholder(node, "ideal", trace)
        assert result == "ideal output"

    def test_ideal_placeholder_without_trace_ideal_falls_back_to_null(self):
        # When trace.ideal is None, _build_placeholder falls back to null behavior.
        node = TraceNode("n", input="x", output="foo")
        trace = AgentTrace(nodes=[node], ideal=None)
        result = _build_placeholder(node, "ideal", trace)
        assert result == ""  # null fallback for string output


# ---------------------------------------------------------------------------
# Candidates and budget
# ---------------------------------------------------------------------------


class TestCandidatesAndBudget:
    def test_candidates_restricts_sweep(self):
        trace = linear_trace()
        deltas = {"a": 0.5, "r": 0.3, "c": 0.2}
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas),
            candidates=["a", "r"],
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        ablated_ids = {e["node_id"] for e in result["effects"]}
        assert "a" in ablated_ids
        assert "r" in ablated_ids
        assert "c" not in ablated_ids

    def test_unknown_candidate_id_raises(self):
        trace = linear_trace()
        with pytest.raises(ValueError, match="unknown span ids"):
            run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=stub_runner,
                judge=judge_from_deltas({}),
                candidates=["a", "does_not_exist"],
                score_range=(0.0, 1.0),
            )

    def test_budget_caps_number_of_runs(self):
        trace = linear_trace()
        call_count = {"n": 0}
        def counting_runner(t, overrides):
            call_count["n"] += 1
            return stub_runner(t, overrides)
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=counting_runner,
            judge=judge_from_deltas({}),
            budget=2,
            score_range=(0.0, 1.0),
        )
        assert call_count["n"] == 2
        assert result["num_candidates"] == 2


# ---------------------------------------------------------------------------
# min_delta threshold
# ---------------------------------------------------------------------------


class TestMinDelta:
    def test_span_below_min_delta_excluded_from_culprits(self):
        trace = linear_trace()
        # Only "a" has delta above 0.05
        deltas = {"a": 0.5, "r": 0.02}  # r below min_delta
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas(deltas),
            score_range=(0.0, 1.0),
            min_delta=0.05,
        )
        culprit_ids = {c.node_id for c in result["culprits"]}
        assert "a" in culprit_ids
        assert "r" not in culprit_ids
        # But "r" should still appear in effects
        effect_ids = {e["node_id"] for e in result["effects"]}
        assert "r" in effect_ids


# ---------------------------------------------------------------------------
# Runner / judge failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    def test_single_runner_failure_is_isolated(self, caplog):
        trace = linear_trace()
        def failing_runner(t: AgentTrace, overrides: dict) -> AgentTrace:
            if "a" in overrides:
                raise RuntimeError("simulated failure for span a")
            return stub_runner(t, overrides)

        with caplog.at_level(logging.WARNING):
            result = run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=failing_runner,
                judge=judge_from_deltas({"r": 0.3}),
                score_range=(0.0, 1.0),
                min_delta=0.0,
            )
        assert result["num_failed"] == 1
        # Other spans still ran
        assert result["num_effects"] == len(trace.nodes)
        # Error recorded in effects
        failed_effects = [e for e in result["effects"] if e.get("error")]
        assert len(failed_effects) == 1
        assert failed_effects[0]["node_id"] == "a"

    def test_total_runner_failure_raises(self):
        trace = linear_trace()
        def always_fail(t, overrides):
            raise RuntimeError("always fails")
        with pytest.raises(AblationError, match="all .* ablation runs failed"):
            run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=always_fail,
                judge=judge_from_deltas({}),
                score_range=(0.0, 1.0),
            )

    def test_runner_returning_non_trace_raises(self):
        trace = linear_trace()
        def bad_runner(t, overrides):
            return "not an AgentTrace"
        with pytest.raises(AblationError, match="expected AgentTrace"):
            run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=bad_runner,
                judge=judge_from_deltas({}),
                score_range=(0.0, 1.0),
            )

    def test_judge_returning_non_numeric_raises(self):
        trace = linear_trace()
        def bad_judge(t):
            return "not a number"
        with pytest.raises(AblationError, match="expected numeric"):
            run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=stub_runner,
                judge=bad_judge,
                score_range=(0.0, 1.0),
            )


# ---------------------------------------------------------------------------
# Placeholder with ideal fallback
# ---------------------------------------------------------------------------


class TestIdealPlaceholder:
    def test_ideal_placeholder_uses_trace_ideal(self, caplog):
        trace = linear_trace()
        trace = AgentTrace(nodes=trace.nodes, ideal="perfect answer")
        result = run_ablation(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            runner=stub_runner,
            judge=judge_from_deltas({}),
            placeholder="ideal",
            score_range=(0.0, 1.0),
            min_delta=0.0,
        )
        # Forced outputs for string nodes should be the ideal
        assert result["placeholder"] == "ideal"

    def test_ideal_falls_back_to_null_with_warning(self, caplog):
        trace = linear_trace()  # no ideal set
        with caplog.at_level(logging.WARNING):
            result = run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=stub_runner,
                judge=judge_from_deltas({}),
                placeholder="ideal",
                score_range=(0.0, 1.0),
            )
        assert result["placeholder"] == "null"
        assert any("falling back to 'null'" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_invalid_placeholder_raises(self):
        trace = linear_trace()
        with pytest.raises(ValueError, match="placeholder must be one of"):
            run_ablation(
                trace=trace,
                score=0.9,
                rubric="Quality.",
                runner=stub_runner,
                judge=judge_from_deltas({}),
                placeholder="random",  # type: ignore[arg-type]
                score_range=(0.0, 1.0),
            )

    def test_zero_score_range_raises(self):
        trace = linear_trace()
        with pytest.raises(ValueError, match="score_range must have hi > lo"):
            run_ablation(
                trace=trace,
                score=0.5,
                rubric="Quality.",
                runner=stub_runner,
                judge=judge_from_deltas({}),
                score_range=(0.5, 0.5),
            )

    def test_empty_trace_raises(self):
        trace = AgentTrace(nodes=[])
        with pytest.raises(ValueError, match="zero nodes"):
            run_ablation(
                trace=trace,
                score=0.5,
                rubric="Quality.",
                runner=stub_runner,
                judge=judge_from_deltas({}),
                score_range=(0.0, 1.0),
            )
