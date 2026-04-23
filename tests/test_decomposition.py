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

"""Tests for decomposition.py — score decomposition attribution method."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aevyra_witness import AgentTrace, TraceNode
from aevyra_witness.trace import KIND_REASON, KIND_TOOL

from aevyra_origin.decomposition import DecompositionError, run_decomposition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def linear_trace() -> AgentTrace:
    return AgentTrace(nodes=[
        TraceNode("classify", input="ticket text", output="billing"),
        TraceNode("retrieve", input="billing", output="policy doc..."),
        TraceNode("answer", input="ticket+docs", output="wrong reply", optimize=True),
    ])


def dag_trace() -> AgentTrace:
    """DAG with two 'plan' nodes sharing a name — keyed by id in aggregation."""
    return AgentTrace(nodes=[
        TraceNode("plan", id="p1", kind=KIND_REASON, prompt_id="planner",
                  step=1, input="q", output="call tools", optimize=True),
        TraceNode("search", id="t1", kind=KIND_TOOL, parent_id="p1",
                  input={"q": "x"}, output="result"),
        TraceNode("plan", id="p2", kind=KIND_REASON, prompt_id="planner",
                  step=2, input="ctx", output="respond", optimize=True),
    ])


def _stub_llm(response: str):
    def call(prompt: str) -> str:
        return response
    return call


def _decomp_response(criteria: list[dict[str, Any]]) -> str:
    return json.dumps({"criteria": criteria})


def _make_criterion(
    criterion: str,
    satisfied: bool,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"criterion": criterion, "satisfied": satisfied, "nodes": nodes}


# ---------------------------------------------------------------------------
# Happy path — linear trace
# ---------------------------------------------------------------------------


class TestRunDecompositionLinear:
    def test_happy_path_single_failed_criterion(self):
        trace = linear_trace()
        resp = _decomp_response([
            _make_criterion("Response is accurate", False, [
                {"node_name": "answer", "contribution": 1.0, "reasoning": "Wrong answer."},
            ]),
        ])
        result = run_decomposition(trace=trace, score=0.3, rubric="Quality rubric.", llm=_stub_llm(resp))
        assert len(result["culprits"]) == 1
        c = result["culprits"][0]
        assert c.node_name == "answer"
        assert c.severity == "primary"  # blame = 1.0/1 = 1.0 ≥ 0.5
        assert abs(c.confidence - 1.0) < 1e-9

    def test_happy_path_multiple_criteria(self):
        trace = linear_trace()
        # Two failed criteria: answer gets blamed for both, retrieve for one.
        resp = _decomp_response([
            _make_criterion("Accurate", False, [
                {"node_name": "answer", "contribution": 0.7, "reasoning": "Wrong."},
                {"node_name": "retrieve", "contribution": 0.3, "reasoning": "Bad docs."},
            ]),
            _make_criterion("Concise", False, [
                {"node_name": "answer", "contribution": 1.0, "reasoning": "Verbose."},
            ]),
        ])
        result = run_decomposition(trace=trace, score=0.3, rubric="Quality.", llm=_stub_llm(resp))
        # answer: (0.7 + 1.0) / 2 = 0.85 → primary
        # retrieve: 0.3 / 2 = 0.15 → minor
        culprit_names = {c.node_name for c in result["culprits"]}
        assert "answer" in culprit_names
        assert "retrieve" in culprit_names
        answer_c = next(c for c in result["culprits"] if c.node_name == "answer")
        assert answer_c.severity == "primary"
        retrieve_c = next(c for c in result["culprits"] if c.node_name == "retrieve")
        assert retrieve_c.severity == "minor"

    def test_all_criteria_satisfied_returns_empty_culprits(self):
        trace = linear_trace()
        resp = _decomp_response([
            _make_criterion("Accurate", True, []),
            _make_criterion("Concise", True, []),
        ])
        result = run_decomposition(trace=trace, score=0.9, rubric="Quality.", llm=_stub_llm(resp))
        assert result["culprits"] == []

    def test_weight_normalization(self):
        trace = linear_trace()
        # Contributions don't sum to 1.0 — should be normalized.
        resp = _decomp_response([
            _make_criterion("Correct", False, [
                {"node_name": "classify", "contribution": 0.2, "reasoning": "Bad."},
                {"node_name": "answer", "contribution": 0.4, "reasoning": "Worse."},
            ]),
        ])
        result = run_decomposition(trace=trace, score=0.2, rubric="Quality.", llm=_stub_llm(resp))
        # After normalization: classify = 0.2/0.6 ≈ 0.333, answer = 0.4/0.6 ≈ 0.667
        total_confidence = sum(c.confidence for c in result["culprits"])
        # Both should be proportionally distributed summing roughly to 1.0
        assert abs(total_confidence - 1.0) < 0.01

    def test_severity_thresholds(self):
        trace = linear_trace()
        # One failed criterion, three contributors — one at each severity tier.
        # Contributions already sum to 1.0 so normalization is a no-op, and
        # with a single failed criterion the per-span blame equals the
        # contribution directly:
        #   classify = 0.55 → primary      (≥ 0.5)
        #   retrieve = 0.30 → contributing (≥ 0.2)
        #   answer   = 0.15 → minor        (> 0, < 0.2)
        resp = _decomp_response([
            _make_criterion("Crit1", False, [
                {"node_name": "classify", "contribution": 0.55, "reasoning": ""},
                {"node_name": "retrieve", "contribution": 0.30, "reasoning": ""},
                {"node_name": "answer",   "contribution": 0.15, "reasoning": ""},
            ]),
        ])
        result = run_decomposition(trace=trace, score=0.2, rubric="Rubric.", llm=_stub_llm(resp))
        by_name = {c.node_name: c for c in result["culprits"]}
        assert by_name["classify"].severity == "primary"      # 0.55 ≥ 0.5
        assert by_name["retrieve"].severity == "contributing" # 0.30 ≥ 0.2
        assert by_name["answer"].severity == "minor"          # 0.15 < 0.2


# ---------------------------------------------------------------------------
# DAG trace — keyed by id, not name
# ---------------------------------------------------------------------------


class TestRunDecompositionDAG:
    def test_repeated_names_keyed_by_id(self):
        trace = dag_trace()
        resp = _decomp_response([
            _make_criterion("Plan quality", False, [
                {"node_id": "p1", "node_name": "plan", "contribution": 0.6, "reasoning": "Step 1 bad."},
                {"node_id": "p2", "node_name": "plan", "contribution": 0.4, "reasoning": "Step 2 bad."},
            ]),
        ])
        result = run_decomposition(trace=trace, score=0.2, rubric="Quality.", llm=_stub_llm(resp))
        # Should produce two separate NodeAttributions (one per node_id)
        assert len(result["culprits"]) == 2
        ids = {c.node_id for c in result["culprits"]}
        assert "p1" in ids
        assert "p2" in ids


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestRunDecompositionErrors:
    def test_ambiguous_name_without_id_raises(self):
        trace = dag_trace()
        resp = _decomp_response([
            _make_criterion("Plan quality", False, [
                {"node_name": "plan", "contribution": 1.0, "reasoning": "Ambiguous."},
            ]),
        ])
        with pytest.raises(DecompositionError, match="ambiguous"):
            run_decomposition(trace=trace, score=0.2, rubric="Rubric.", llm=_stub_llm(resp))

    def test_unknown_node_name_raises(self):
        trace = linear_trace()
        resp = _decomp_response([
            _make_criterion("Phantom", False, [
                {"node_name": "ghost_node", "contribution": 1.0, "reasoning": "Nope."},
            ]),
        ])
        with pytest.raises(DecompositionError):
            run_decomposition(trace=trace, score=0.2, rubric="Rubric.", llm=_stub_llm(resp))

    def test_unparseable_response_raises(self):
        trace = linear_trace()
        with pytest.raises(DecompositionError, match="not parseable JSON"):
            run_decomposition(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm("garbage"))

    def test_non_boolean_satisfied_raises(self):
        # _coerce_bool permissively accepts "yes"/"no"/"true"/"false"/"0"/"1"
        # (LLMs often return these in JSON), but uncoerceable values like
        # "maybe" or arbitrary numbers should raise.
        trace = linear_trace()
        resp = json.dumps({"criteria": [
            {"criterion": "Accurate", "satisfied": "maybe", "nodes": []}
        ]})
        with pytest.raises(DecompositionError, match="non-boolean"):
            run_decomposition(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))
