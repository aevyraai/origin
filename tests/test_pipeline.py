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

"""Tests for diagnose_pipeline — the turnkey Origin entry point."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aevyra_witness import AgentTrace
from aevyra_witness.runtime import span

from aevyra_origin import PipelineError, diagnose_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stub_llm_critic_response(culprits: list[dict[str, Any]]) -> str:
    return json.dumps({
        "summary": "stub summary",
        "culprits": culprits,
    })


def _stub_llm(culprits: list[dict[str, Any]]):
    """Build an LLM stub that emits a critic response citing ``culprits``."""
    payload = _stub_llm_critic_response(culprits)

    def llm(_prompt: str) -> str:
        return payload

    return llm


def _constant_llm(payload: str):
    def llm(_prompt: str) -> str:
        return payload

    return llm


def _always(score: float):
    def judge(_trace: AgentTrace) -> float:
        return score

    return judge


# A minimal instrumented pipeline.

@span("classify")
def _classify(text: str) -> str:
    return "billing"


@span("retrieve")
def _retrieve(topic: str) -> list[str]:
    return ["doc1", "doc2"]


@span("answer", optimize=True, prompt_id="answer_v1")
def _answer(q: str, docs: list[str]) -> str:
    return "wrong answer"


def _pipeline(q: str) -> str:
    t = _classify(q)
    d = _retrieve(t)
    return _answer(q, d)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestDiagnosePipelineHappyPath:
    def test_end_to_end_with_stub_llm_and_judge(self):
        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "primary",
             "confidence": 0.9, "reasoning": "answer is wrong"},
        ])

        result = diagnose_pipeline(
            _pipeline, "how do I refund?",
            judge=_always(0.3), rubric="Accurate and concise.", llm=llm,
            method="critic",
        )

        assert result.score == 0.3
        assert len(result.culprits) == 1
        assert result.culprits[0].node_name == "answer"
        assert result.culprits[0].severity == "primary"

    def test_captured_trace_and_output_surfaced_in_raw(self):
        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "primary",
             "confidence": 0.9, "reasoning": "x"},
        ])

        result = diagnose_pipeline(
            _pipeline, "q",
            judge=_always(0.5), rubric="r", llm=llm, method="critic",
        )

        assert result.raw["pipeline_output"] == "wrong answer"
        captured = result.raw["captured_trace"]
        assert isinstance(captured, dict)
        assert len(captured["nodes"]) == 3
        names = [n["name"] for n in captured["nodes"]]
        assert names == ["classify", "retrieve", "answer"]

    def test_ideal_propagates_to_trace(self):
        observed = {}

        def judge(t: AgentTrace) -> float:
            observed["ideal"] = t.ideal
            return 0.4

        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "primary",
             "confidence": 0.8, "reasoning": "x"},
        ])

        diagnose_pipeline(
            _pipeline, "q",
            judge=judge, rubric="r", llm=llm,
            ideal="the correct reply", method="critic",
        )
        assert observed["ideal"] == "the correct reply"

    def test_trace_metadata_propagates(self):
        observed = {}

        def judge(t: AgentTrace) -> float:
            observed["metadata"] = dict(t.metadata)
            return 0.5

        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "primary",
             "confidence": 0.8, "reasoning": "x"},
        ])

        diagnose_pipeline(
            _pipeline, "q",
            judge=judge, rubric="r", llm=llm,
            trace_metadata={"model": "gpt-4o", "run_id": "abc"},
            method="critic",
        )
        assert observed["metadata"] == {"model": "gpt-4o", "run_id": "abc"}

    def test_kwargs_forwarded_to_pipeline(self):
        seen = {}

        @span("fn")
        def fn(a, b, *, c):
            seen.update({"a": a, "b": b, "c": c})
            return "out"

        llm = _stub_llm([
            {"node_name": "fn", "node_id": "n0", "severity": "primary",
             "confidence": 0.5, "reasoning": "x"},
        ])

        diagnose_pipeline(
            fn, 1, 2,
            c=3,
            judge=_always(0.5), rubric="r", llm=llm, method="critic",
        )
        assert seen == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# Judge return-value coercion
# ---------------------------------------------------------------------------


class TestJudgeReturn:
    def test_judge_can_return_int(self):
        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "minor",
             "confidence": 0.2, "reasoning": "x"},
        ])
        result = diagnose_pipeline(
            _pipeline, "q",
            judge=lambda _t: 1, rubric="r", llm=llm, method="critic",
        )
        assert result.score == 1.0

    def test_judge_can_return_score_result_like_object(self):
        # Verdict-style ScoreResult duck-type.
        class Result:
            score = 0.77

        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "contributing",
             "confidence": 0.3, "reasoning": "x"},
        ])
        result = diagnose_pipeline(
            _pipeline, "q",
            judge=lambda _t: Result(), rubric="r", llm=llm, method="critic",
        )
        assert result.score == 0.77

    def test_judge_returning_nan_raises(self):
        with pytest.raises(PipelineError, match="NaN"):
            diagnose_pipeline(
                _pipeline, "q",
                judge=lambda _t: float("nan"), rubric="r",
                llm=_constant_llm("{}"), method="critic",
            )

    def test_judge_returning_garbage_raises(self):
        with pytest.raises(PipelineError, match="must return a float"):
            diagnose_pipeline(
                _pipeline, "q",
                judge=lambda _t: object(), rubric="r",
                llm=_constant_llm("{}"), method="critic",
            )

    def test_judge_exception_is_wrapped(self):
        def exploding_judge(_t: AgentTrace) -> float:
            raise RuntimeError("judge broke")

        with pytest.raises(PipelineError, match="judge raised"):
            diagnose_pipeline(
                _pipeline, "q",
                judge=exploding_judge, rubric="r",
                llm=_constant_llm("{}"), method="critic",
            )


# ---------------------------------------------------------------------------
# Empty trace / validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_uninstrumented_pipeline_raises_pipeline_error(self):
        # No @span anywhere — pipeline runs fine but no spans are emitted.
        def bare(q):
            return q.upper()

        with pytest.raises(PipelineError, match="empty trace"):
            diagnose_pipeline(
                bare, "q",
                judge=_always(0.3), rubric="r",
                llm=_constant_llm("{}"), method="critic",
            )

    def test_non_callable_pipeline_raises(self):
        with pytest.raises(ValueError, match="pipeline must be callable"):
            diagnose_pipeline(
                "not a callable",  # type: ignore[arg-type]
                judge=_always(0.3), rubric="r", llm=_constant_llm("{}"),
            )

    def test_non_callable_judge_raises(self):
        with pytest.raises(ValueError, match="judge must be callable"):
            diagnose_pipeline(
                _pipeline, "q",
                judge="nope",  # type: ignore[arg-type]
                rubric="r", llm=_constant_llm("{}"),
            )


# ---------------------------------------------------------------------------
# Ablation integration
# ---------------------------------------------------------------------------


class TestAblationIntegration:
    def test_runner_enables_ablation(self):
        # A runner that replays by rebuilding the trace with overrides applied.
        def runner(original: AgentTrace, overrides: dict) -> AgentTrace:
            # Simple runner: copy nodes, replace outputs per overrides.
            from aevyra_witness import TraceNode
            new_nodes = []
            for n in original.nodes:
                if n.id in overrides:
                    new_nodes.append(TraceNode(
                        name=n.name, input=n.input, output=overrides[n.id],
                        id=n.id, parent_id=n.parent_id, kind=n.kind,
                        optimize=n.optimize, prompt_id=n.prompt_id,
                    ))
                else:
                    new_nodes.append(n)
            return AgentTrace(nodes=new_nodes, ideal=original.ideal)

        # Judge: score is 0.1 if answer span is empty, else 0.5.
        def judge(t: AgentTrace) -> float:
            ans = next((n for n in t.nodes if n.name == "answer"), None)
            if ans and (ans.output == "" or ans.output is None):
                return 0.1
            return 0.5

        llm = _stub_llm([
            {"node_name": "answer", "node_id": "n2", "severity": "primary",
             "confidence": 0.8, "reasoning": "x"},
        ])

        result = diagnose_pipeline(
            _pipeline, "q",
            judge=judge, rubric="r", llm=llm,
            runner=runner, method="all",
        )

        assert "ablation" in result.raw
        # Ablating the answer span should cause a big delta.
        effects = result.raw["ablation"]["effects"]
        assert any(e["node_id"] == "n2" for e in effects)
