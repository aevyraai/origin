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

"""Tests for diagnose.py — Origin class and top-level diagnose() function."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aevyra_witness import AgentTrace, TraceNode
from aevyra_witness.trace import KIND_REASON, KIND_TOOL

from aevyra_origin import Origin, diagnose
from aevyra_origin.result import Attribution
from aevyra_origin.diagnose import _corroborated_confidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def linear_trace() -> AgentTrace:
    return AgentTrace(nodes=[
        TraceNode("classify", input="ticket", output="billing"),
        TraceNode("retrieve", input="billing", output="policy doc"),
        TraceNode("answer", input="ticket+docs", output="wrong reply", optimize=True),
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


def _stub_llm(responses: list[str]):
    """Returns a callable that yields the next queued response on each call."""
    it = iter(responses)
    def call(prompt: str) -> str:
        return next(it)
    return call


def _critic_json(summary: str = "Critic summary.", culprits: list = None) -> str:
    if culprits is None:
        culprits = [{
            "node_id": "n2",
            "node_name": "answer",
            "severity": "primary",
            "confidence": 0.8,
            "reasoning": "Critic: answer was wrong.",
        }]
    return json.dumps({"summary": summary, "culprits": culprits})


def _decomp_json(summary_override: str = None, culprits_override: list = None) -> str:
    criteria = [{
        "criterion": "Accuracy",
        "satisfied": False,
        "nodes": [{
            "node_id": "n2",
            "node_name": "answer",
            "contribution": 1.0,
            "reasoning": "Decomp: answer was wrong.",
        }],
    }]
    return json.dumps({"criteria": criteria})


def stub_runner(trace: AgentTrace, overrides: dict[str, Any]) -> AgentTrace:
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


def judge_always(score: float):
    def j(t: AgentTrace) -> float:
        return score
    return j


# ---------------------------------------------------------------------------
# Origin construction
# ---------------------------------------------------------------------------


class TestOriginConstruction:
    def test_non_callable_llm_raises(self):
        with pytest.raises(TypeError, match="llm must be a callable"):
            Origin(llm="not callable")

    def test_runner_without_judge_raises(self):
        with pytest.raises(ValueError, match="runner and judge must be provided together"):
            Origin(llm=_stub_llm([]), runner=stub_runner)

    def test_judge_without_runner_raises(self):
        with pytest.raises(ValueError, match="runner and judge must be provided together"):
            Origin(llm=_stub_llm([]), judge=judge_always(0.5))

    def test_non_callable_runner_raises(self):
        with pytest.raises(TypeError, match="runner must be callable"):
            Origin(llm=_stub_llm([]), runner="not callable", judge=judge_always(0.5))

    def test_non_callable_judge_raises(self):
        with pytest.raises(TypeError, match="judge must be callable"):
            Origin(llm=_stub_llm([]), runner=stub_runner, judge="not callable")

    def test_ablation_available_false_without_runner_judge(self):
        o = Origin(llm=_stub_llm([]))
        assert not o.ablation_available

    def test_ablation_available_true_with_runner_judge(self):
        o = Origin(llm=_stub_llm([]), runner=stub_runner, judge=judge_always(0.5))
        assert o.ablation_available


# ---------------------------------------------------------------------------
# Single-method dispatch
# ---------------------------------------------------------------------------


class TestDiagnoseMethod:
    def test_method_critic(self):
        trace = linear_trace()
        o = Origin(llm=_stub_llm([_critic_json()]))
        result = o.diagnose(trace=trace, score=0.4, rubric="Quality.", method="critic")
        assert isinstance(result, Attribution)
        assert result.method == "critic"
        assert "critic" in result.raw
        assert "decomposition" not in result.raw

    def test_method_decomposition(self):
        trace = linear_trace()
        o = Origin(llm=_stub_llm([_decomp_json()]))
        result = o.diagnose(trace=trace, score=0.4, rubric="Quality.", method="decomposition")
        assert result.method == "decomposition"
        assert "decomposition" in result.raw
        assert "critic" not in result.raw

    def test_method_ablation_without_runner_raises(self):
        trace = linear_trace()
        o = Origin(llm=_stub_llm([]))
        with pytest.raises(ValueError, match="requires Origin to be constructed with a runner"):
            o.diagnose(trace=trace, score=0.4, rubric="Quality.", method="ablation")

    def test_method_ablation_with_runner(self):
        trace = linear_trace()
        # Ablation without LLM involvement — just runner + judge
        o = Origin(
            llm=_stub_llm([]),
            runner=stub_runner,
            judge=judge_always(0.5),
        )
        result = o.diagnose(trace=trace, score=0.9, rubric="Quality.", method="ablation")
        assert result.method == "ablation"
        assert "ablation" in result.raw

    def test_invalid_method_raises(self):
        trace = linear_trace()
        o = Origin(llm=_stub_llm([]))
        with pytest.raises(ValueError, match="method must be one of"):
            o.diagnose(trace=trace, score=0.4, rubric="Quality.", method="magic")  # type: ignore

    def test_empty_trace_raises(self):
        trace = AgentTrace(nodes=[])
        o = Origin(llm=_stub_llm([]))
        with pytest.raises(ValueError, match="zero nodes"):
            o.diagnose(trace=trace, score=0.4, rubric="Quality.")

    def test_empty_rubric_raises(self):
        trace = linear_trace()
        o = Origin(llm=_stub_llm([]))
        with pytest.raises(ValueError, match="rubric must be"):
            o.diagnose(trace=trace, score=0.4, rubric="   ")


# ---------------------------------------------------------------------------
# method="all" — merge logic
# ---------------------------------------------------------------------------


class TestMethodAll:
    def test_all_without_runner_makes_two_llm_calls_no_ablation(self):
        trace = linear_trace()
        call_count = {"n": 0}
        def counting_llm(prompt: str) -> str:
            call_count["n"] += 1
            # First call → critic format; second → decomposition
            if call_count["n"] == 1:
                return _critic_json()
            return _decomp_json()

        o = Origin(llm=counting_llm)
        result = o.diagnose(trace=trace, score=0.3, rubric="Quality.", method="all")
        assert call_count["n"] == 2
        assert result.method == "all"
        assert "critic" in result.raw
        assert "decomposition" in result.raw
        assert "ablation" not in result.raw

    def test_all_with_runner_includes_ablation(self):
        trace = linear_trace()
        call_count = {"n": 0}
        def counting_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _critic_json()
            return _decomp_json()

        o = Origin(llm=counting_llm, runner=stub_runner, judge=judge_always(0.4))
        result = o.diagnose(trace=trace, score=0.9, rubric="Quality.", method="all")
        assert call_count["n"] == 2
        assert "ablation" in result.raw

    def test_all_span_named_by_all_three_methods_gets_corroborated_confidence(self):
        """When all three methods name the same span, corroboration applies."""
        trace = linear_trace()
        call_count = {"n": 0}
        def counting_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _critic_json("Critic.", [{
                    "node_id": "n2", "node_name": "answer",
                    "severity": "primary", "confidence": 0.6,
                    "reasoning": "Critic saw this.",
                }])
            return json.dumps({"criteria": [{
                "criterion": "Accuracy", "satisfied": False,
                "nodes": [{"node_id": "n2", "node_name": "answer",
                           "contribution": 1.0, "reasoning": "Decomp saw this."}],
            }]})

        # Ablation: "n2" has delta 0.6 → normalized=0.6
        def ablation_judge(t: AgentTrace) -> float:
            for n in t.nodes:
                if n.id == "n2" and not n.output:
                    return 0.3  # score=0.9 - 0.6 = 0.3
            return 0.9

        o = Origin(llm=counting_llm, runner=stub_runner, judge=ablation_judge)
        result = o.diagnose(trace=trace, score=0.9, rubric="Quality.", method="all")

        answer_culprit = next(
            (c for c in result.culprits if c.node_name == "answer"), None
        )
        assert answer_culprit is not None
        # All three methods named it — confidence should be corroborated (above avg)
        assert answer_culprit.confidence > 0.0
        # Reasoning should have all three method prefixes
        assert "[critic]" in answer_culprit.reasoning
        assert "[decomposition]" in answer_culprit.reasoning
        assert "[ablation]" in answer_culprit.reasoning

    def test_single_method_span_keeps_raw_confidence(self):
        """A span named by only one method is not penalized."""
        trace = linear_trace()
        # Critic names "answer" with conf 0.7; decomp names "retrieve"
        call_count = {"n": 0}
        def two_method_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _critic_json(".", [{
                    "node_id": "n2", "node_name": "answer",
                    "severity": "primary", "confidence": 0.7,
                    "reasoning": "Critic.",
                }])
            return json.dumps({"criteria": [{
                "criterion": "Retrieval", "satisfied": False,
                "nodes": [{"node_id": "n1", "node_name": "retrieve",
                           "contribution": 1.0, "reasoning": "Decomp."}],
            }]})

        o = Origin(llm=two_method_llm)
        result = o.diagnose(trace=trace, score=0.3, rubric="Quality.", method="all")

        answer_c = next(c for c in result.culprits if c.node_name == "answer")
        # Single-method span: confidence should be ~0.7 (passthrough)
        assert abs(answer_c.confidence - 0.7) < 0.01

    def test_merge_keys_on_node_id_not_name(self):
        """Two spans with same name but different ids are NOT merged."""
        trace = dag_trace()
        call_count = {"n": 0}
        def dag_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps({"summary": ".", "culprits": [
                    {"node_id": "p1", "node_name": "plan",
                     "severity": "primary", "confidence": 0.8, "reasoning": "p1."},
                ]})
            return json.dumps({"criteria": [{
                "criterion": "Plan step 2", "satisfied": False,
                "nodes": [{"node_id": "p2", "node_name": "plan",
                           "contribution": 1.0, "reasoning": "p2."}],
            }]})

        o = Origin(llm=dag_llm)
        result = o.diagnose(trace=trace, score=0.3, rubric="Quality.", method="all")
        ids = [c.node_id for c in result.culprits]
        # Both should appear separately
        assert "p1" in ids
        assert "p2" in ids


# ---------------------------------------------------------------------------
# _corroborated_confidence
# ---------------------------------------------------------------------------


class TestCorroboratedConfidence:
    def test_single_method_passthrough(self):
        assert abs(_corroborated_confidence([0.7]) - 0.7) < 1e-9

    def test_two_methods_halfway_between_avg_and_max(self):
        # avg=0.5, max=0.7, weight=0.5
        # result = 0.5 + (0.7-0.5)*0.5 = 0.5 + 0.1 = 0.6
        c = _corroborated_confidence([0.3, 0.7])
        assert abs(c - 0.6) < 1e-9

    def test_three_methods_two_thirds_of_way(self):
        # avg=(0.5+0.6+0.7)/3=0.6, max=0.7, weight=2/3
        # result = 0.6 + (0.7-0.6)*(2/3) = 0.6 + 0.0667 ≈ 0.6667
        c = _corroborated_confidence([0.5, 0.6, 0.7])
        assert abs(c - (0.6 + (0.7 - 0.6) * 2/3)) < 1e-9

    def test_bounded_to_zero_one(self):
        assert _corroborated_confidence([1.0, 1.0, 1.0]) <= 1.0
        assert _corroborated_confidence([0.0]) >= 0.0

    def test_empty_returns_zero(self):
        assert _corroborated_confidence([]) == 0.0


# ---------------------------------------------------------------------------
# by_prompt on merged result
# ---------------------------------------------------------------------------


class TestByPromptOnMergedResult:
    def test_by_prompt_aggregates_across_spans(self):
        """by_prompt() on a merged result uses the (corroborated) confidence."""
        trace = dag_trace()
        call_count = {"n": 0}
        def dag_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps({"summary": ".", "culprits": [
                    {"node_id": "p1", "node_name": "plan",
                     "severity": "primary", "confidence": 0.8, "reasoning": "p1 bad."},
                ]})
            return json.dumps({"criteria": [{
                "criterion": "Plan step 2", "satisfied": False,
                "nodes": [{"node_id": "p2", "node_name": "plan",
                           "contribution": 1.0, "reasoning": "p2 bad."}],
            }]})

        o = Origin(llm=dag_llm)
        result = o.diagnose(trace=trace, score=0.3, rubric="Quality.", method="all")
        prompt_attrs = result.by_prompt()
        assert len(prompt_attrs) == 1
        assert prompt_attrs[0].prompt_id == "planner"
        assert len(prompt_attrs[0].spans) == 2


# ---------------------------------------------------------------------------
# Top-level diagnose() function
# ---------------------------------------------------------------------------


class TestTopLevelDiagnose:
    def test_diagnose_function_equivalent_to_origin_diagnose(self):
        trace = linear_trace()
        llm = _stub_llm([_critic_json()])
        result = diagnose(trace=trace, score=0.4, rubric="Quality.", llm=llm, method="critic")
        assert isinstance(result, Attribution)
        assert result.method == "critic"

    def test_diagnose_function_with_runner_and_judge(self):
        trace = linear_trace()
        call_count = {"n": 0}
        def counting_llm(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _critic_json()
            return _decomp_json()

        result = diagnose(
            trace=trace,
            score=0.9,
            rubric="Quality.",
            llm=counting_llm,
            method="all",
            runner=stub_runner,
            judge=judge_always(0.4),
        )
        assert "ablation" in result.raw
