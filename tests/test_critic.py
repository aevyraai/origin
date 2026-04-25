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

"""Tests for critic.py — LLM-as-critic attribution method."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from aevyra_witness import AgentTrace, TraceNode
from aevyra_witness.trace import KIND_REASON, KIND_TOOL

from aevyra_origin.critic import CriticError, run_critic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def linear_trace() -> AgentTrace:
    return AgentTrace(
        nodes=[
            TraceNode("classify", input="ticket text", output="billing"),
            TraceNode("retrieve", input="billing", output="policy doc..."),
            TraceNode("answer", input="ticket+docs", output="wrong reply", optimize=True),
        ]
    )


def dag_trace() -> AgentTrace:
    return AgentTrace(
        nodes=[
            TraceNode(
                "plan",
                id="p1",
                kind=KIND_REASON,
                prompt_id="planner",
                step=1,
                input="q",
                output="call tools",
                optimize=True,
            ),
            TraceNode(
                "search", id="t1", kind=KIND_TOOL, parent_id="p1", input={"q": "x"}, output="result"
            ),
            TraceNode(
                "plan",
                id="p2",
                kind=KIND_REASON,
                prompt_id="planner",
                step=2,
                input="ctx",
                output="respond",
                optimize=True,
            ),
        ]
    )


def _stub_llm(response: str):
    """Returns an LLMFn that always returns ``response``."""

    def call(prompt: str) -> str:
        return response

    return call


def _critic_response(summary: str, culprits: list[dict[str, Any]]) -> str:
    return json.dumps({"summary": summary, "culprits": culprits})


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestRunCriticLinear:
    def test_culprit_cited_by_name_only(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="The answer node gave a wrong reply.",
            culprits=[
                {
                    "node_name": "answer",
                    "severity": "primary",
                    "confidence": 0.9,
                    "reasoning": "The answer was incorrect.",
                }
            ],
        )
        result = run_critic(trace=trace, score=0.3, rubric="Quality rubric.", llm=_stub_llm(resp))
        assert result["summary"] == "The answer node gave a wrong reply."
        assert len(result["culprits"]) == 1
        c = result["culprits"][0]
        assert c.node_name == "answer"
        assert c.severity == "primary"
        assert c.confidence == 0.9

    def test_culprit_cited_by_id(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Retrieve failed.",
            culprits=[
                {
                    "node_id": "n1",
                    "node_name": "retrieve",
                    "severity": "contributing",
                    "confidence": 0.6,
                    "reasoning": "Retrieved wrong doc.",
                }
            ],
        )
        result = run_critic(trace=trace, score=0.4, rubric="Quality rubric.", llm=_stub_llm(resp))
        c = result["culprits"][0]
        assert c.node_id == "n1"
        assert c.node_name == "retrieve"
        assert c.prompt_id is None  # retrieve has no prompt_id

    def test_sorted_by_confidence_descending(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Multiple failures.",
            culprits=[
                {
                    "node_name": "classify",
                    "severity": "contributing",
                    "confidence": 0.4,
                    "reasoning": "Misclassified.",
                },
                {
                    "node_name": "answer",
                    "severity": "primary",
                    "confidence": 0.9,
                    "reasoning": "Wrong answer.",
                },
            ],
        )
        result = run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))
        assert result["culprits"][0].confidence >= result["culprits"][1].confidence

    def test_empty_culprits_list_is_valid(self):
        trace = linear_trace()
        resp = _critic_response(summary="No failure detected.", culprits=[])
        result = run_critic(trace=trace, score=0.8, rubric="Rubric.", llm=_stub_llm(resp))
        assert result["culprits"] == []
        assert result["summary"] == "No failure detected."

    def test_raw_llm_response_in_result(self):
        trace = linear_trace()
        raw_resp = _critic_response("OK.", culprits=[])
        result = run_critic(trace=trace, score=0.8, rubric="Rubric.", llm=_stub_llm(raw_resp))
        assert result["raw"] == raw_resp

    def test_fenced_llm_response_parses(self):
        trace = linear_trace()
        inner = _critic_response("Fenced.", culprits=[])
        fenced = f"```json\n{inner}\n```\n\nHope this helps."
        result = run_critic(trace=trace, score=0.5, rubric="Rubric.", llm=_stub_llm(fenced))
        assert result["summary"] == "Fenced."

    def test_out_of_range_confidence_is_clamped(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Clamped.",
            culprits=[
                {
                    "node_name": "answer",
                    "severity": "primary",
                    "confidence": 1.5,  # over range
                    "reasoning": "Too confident.",
                }
            ],
        )
        result = run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))
        assert result["culprits"][0].confidence == 1.0

    def test_prompt_id_carried_from_span(self):
        # Build a trace where the node has a prompt_id.
        trace = AgentTrace(
            nodes=[
                TraceNode("plan", id="p1", prompt_id="planner", input="q", output="o"),
            ]
        )
        resp = _critic_response(
            summary="Plan failed.",
            culprits=[
                {
                    "node_id": "p1",
                    "node_name": "plan",
                    "severity": "primary",
                    "confidence": 0.8,
                    "reasoning": "Planning was wrong.",
                }
            ],
        )
        result = run_critic(trace=trace, score=0.2, rubric="Rubric.", llm=_stub_llm(resp))
        assert result["culprits"][0].prompt_id == "planner"


# ---------------------------------------------------------------------------
# DAG trace tests
# ---------------------------------------------------------------------------


class TestRunCriticDAG:
    def test_culprit_cited_by_node_id_in_dag(self):
        trace = dag_trace()
        resp = _critic_response(
            summary="First plan step failed.",
            culprits=[
                {
                    "node_id": "p1",
                    "node_name": "plan",
                    "severity": "primary",
                    "confidence": 0.85,
                    "reasoning": "Wrong tool selection.",
                }
            ],
        )
        result = run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))
        c = result["culprits"][0]
        assert c.node_id == "p1"
        assert c.node_name == "plan"
        assert c.prompt_id == "planner"

    def test_ambiguous_name_without_id_raises(self):
        trace = dag_trace()  # has two "plan" nodes
        resp = _critic_response(
            summary="Plan failed.",
            culprits=[
                {
                    "node_name": "plan",  # ambiguous — no id
                    "severity": "primary",
                    "confidence": 0.8,
                    "reasoning": "Bad plan.",
                }
            ],
        )
        with pytest.raises(CriticError, match="ambiguous node_name"):
            run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestRunCriticErrors:
    def test_unknown_node_id_raises(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Unknown.",
            culprits=[
                {
                    "node_id": "nonexistent_id",
                    "node_name": "ghost",
                    "severity": "primary",
                    "confidence": 0.5,
                    "reasoning": "Ghost node.",
                }
            ],
        )
        with pytest.raises(CriticError, match="unknown node_id"):
            run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))

    def test_unknown_node_name_raises(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Unknown.",
            culprits=[
                {
                    "node_name": "nonexistent_node",
                    "severity": "primary",
                    "confidence": 0.5,
                    "reasoning": "Doesn't exist.",
                }
            ],
        )
        with pytest.raises(CriticError, match="unknown node_name"):
            run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))

    def test_invalid_severity_raises(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Bad sev.",
            culprits=[
                {
                    "node_name": "answer",
                    "severity": "catastrophic",  # invalid
                    "confidence": 0.7,
                    "reasoning": "Very bad.",
                }
            ],
        )
        with pytest.raises(CriticError, match="invalid severity"):
            run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))

    def test_non_numeric_confidence_raises(self):
        trace = linear_trace()
        resp = _critic_response(
            summary="Bad conf.",
            culprits=[
                {
                    "node_name": "answer",
                    "severity": "primary",
                    "confidence": "high",  # non-numeric
                    "reasoning": "Bad.",
                }
            ],
        )
        with pytest.raises(CriticError, match="non-numeric confidence"):
            run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))

    def test_unparseable_response_raises(self):
        trace = linear_trace()
        with pytest.raises(CriticError, match="not parseable JSON"):
            run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm("not json at all"))

    def test_name_id_mismatch_logs_warning_id_wins(self, caplog):
        trace = linear_trace()
        resp = _critic_response(
            summary="Mismatch.",
            culprits=[
                {
                    "node_id": "n0",  # this is "classify"
                    "node_name": "answer",  # wrong name for this id
                    "severity": "primary",
                    "confidence": 0.7,
                    "reasoning": "Mismatch test.",
                }
            ],
        )
        with caplog.at_level(logging.WARNING, logger="aevyra_origin.critic"):
            result = run_critic(trace=trace, score=0.3, rubric="Rubric.", llm=_stub_llm(resp))
        # id wins — node_name should be "classify" (the actual name at n0)
        assert result["culprits"][0].node_id == "n0"
        assert result["culprits"][0].node_name == "classify"
        assert any("does not match" in r.message for r in caplog.records)
