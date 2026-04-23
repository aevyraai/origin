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

"""Tests for aevyra_origin.judges — Verdict adapter + default extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aevyra_witness import AgentTrace, TraceNode

from aevyra_origin.judges import (
    default_messages_from_trace,
    default_response_from_trace,
    judge_from_verdict,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeScoreResult:
    """Duck-types Verdict's ScoreResult."""
    score: float
    reasoning: str | None = None


class _FakeMetric:
    """Duck-types Verdict's Metric.score(response, ideal, messages)."""

    def __init__(self, score: float = 0.5):
        self._score = score
        self.calls: list[dict[str, Any]] = []

    def score(self, response, ideal, messages):
        self.calls.append({"response": response, "ideal": ideal, "messages": messages})
        return _FakeScoreResult(score=self._score, reasoning="stub")


def _make_trace(
    *,
    input_: Any = "user question",
    output: Any = "pipeline output",
    ideal: str | None = None,
) -> AgentTrace:
    """Build a simple 1-span AgentTrace for extractor testing."""
    return AgentTrace(
        nodes=[TraceNode(name="root", input=input_, output=output)],
        ideal=ideal,
    )


def _make_empty_trace() -> AgentTrace:
    return AgentTrace(nodes=[])


# ---------------------------------------------------------------------------
# default_response_from_trace
# ---------------------------------------------------------------------------


class TestDefaultResponseFromTrace:
    def test_string_output_passthrough(self):
        t = _make_trace(output="hello world")
        assert default_response_from_trace(t) == "hello world"

    def test_none_output_becomes_empty_string(self):
        t = _make_trace(output=None)
        assert default_response_from_trace(t) == ""

    def test_non_string_output_json_encoded(self):
        t = _make_trace(output={"answer": "refund", "confidence": 0.9})
        out = default_response_from_trace(t)
        # JSON-encoded dict, no repr() fallback
        assert '"answer"' in out
        assert '"refund"' in out

    def test_list_output_json_encoded(self):
        t = _make_trace(output=["a", "b", "c"])
        assert default_response_from_trace(t) == '["a", "b", "c"]'

    def test_unserializable_output_falls_back_to_repr(self):
        class Weird:
            def __repr__(self) -> str:
                return "<WEIRD>"

        t = _make_trace(output=Weird())
        out = default_response_from_trace(t)
        # json.dumps with default=str handles non-serializable; repr() fallback
        # only kicks in if default=str also fails.
        assert "WEIRD" in out

    def test_last_root_wins_when_multiple_roots(self):
        t = AgentTrace(nodes=[
            TraceNode(name="a", input="q", output="first"),
            TraceNode(name="b", input="q", output="second"),
        ])
        assert default_response_from_trace(t) == "second"

    def test_empty_trace_raises(self):
        with pytest.raises(ValueError, match="no root span"):
            default_response_from_trace(_make_empty_trace())


# ---------------------------------------------------------------------------
# default_messages_from_trace
# ---------------------------------------------------------------------------


class TestDefaultMessagesFromTrace:
    def test_string_input_wrapped_as_user_message(self):
        t = _make_trace(input_="what's the refund policy?")
        msgs = default_messages_from_trace(t)
        assert msgs == [{"role": "user", "content": "what's the refund policy?"}]

    def test_non_string_input_returns_none(self):
        t = _make_trace(input_={"query": "refund"})
        assert default_messages_from_trace(t) is None

    def test_empty_trace_returns_none(self):
        assert default_messages_from_trace(_make_empty_trace()) is None

    def test_first_root_wins(self):
        t = AgentTrace(nodes=[
            TraceNode(name="a", input="first", output="x"),
            TraceNode(name="b", input="second", output="y"),
        ])
        msgs = default_messages_from_trace(t)
        assert msgs == [{"role": "user", "content": "first"}]


# ---------------------------------------------------------------------------
# judge_from_verdict — core behaviour
# ---------------------------------------------------------------------------


class TestJudgeFromVerdictCore:
    def test_returns_float_from_score_result(self):
        metric = _FakeMetric(score=0.73)
        judge = judge_from_verdict(metric)
        t = _make_trace()
        assert judge(t) == 0.73

    def test_passes_response_ideal_messages_to_metric(self):
        metric = _FakeMetric(score=1.0)
        judge = judge_from_verdict(metric)
        t = _make_trace(input_="hello", output="world", ideal="expected")
        judge(t)
        call = metric.calls[0]
        assert call["response"] == "world"
        assert call["ideal"] == "expected"
        assert call["messages"] == [{"role": "user", "content": "hello"}]

    def test_trace_ideal_used_by_default(self):
        metric = _FakeMetric(score=0.0)
        judge = judge_from_verdict(metric)
        t = _make_trace(ideal="reference")
        judge(t)
        assert metric.calls[0]["ideal"] == "reference"

    def test_explicit_ideal_overrides_trace_ideal(self):
        metric = _FakeMetric(score=0.0)
        judge = judge_from_verdict(metric, ideal="override")
        t = _make_trace(ideal="from-trace")
        judge(t)
        assert metric.calls[0]["ideal"] == "override"

    def test_explicit_ideal_works_when_trace_ideal_is_none(self):
        metric = _FakeMetric(score=0.0)
        judge = judge_from_verdict(metric, ideal="override")
        t = _make_trace(ideal=None)
        judge(t)
        assert metric.calls[0]["ideal"] == "override"

    def test_metric_returning_plain_float(self):
        """Metric that returns a bare number (no ScoreResult object) still works."""
        class NumericMetric:
            def score(self, response, ideal, messages):
                return 0.42

        judge = judge_from_verdict(NumericMetric())
        assert judge(_make_trace()) == 0.42

    def test_metric_returning_int(self):
        class IntMetric:
            def score(self, response, ideal, messages):
                return 1

        judge = judge_from_verdict(IntMetric())
        assert judge(_make_trace()) == 1.0

    def test_non_numeric_score_raises_type_error(self):
        class BadMetric:
            def score(self, response, ideal, messages):
                return _FakeScoreResult(score="not a number")  # type: ignore[arg-type]

        judge = judge_from_verdict(BadMetric())
        with pytest.raises(TypeError, match="not coercible to float"):
            judge(_make_trace())


# ---------------------------------------------------------------------------
# Custom extractors
# ---------------------------------------------------------------------------


class TestCustomExtractors:
    def test_custom_extract_response(self):
        # Pull from a specific named span instead of last root.
        metric = _FakeMetric(score=1.0)

        def extract(trace: AgentTrace) -> str:
            answer = next(n for n in trace.nodes if n.name == "answer")
            return str(answer.output)

        judge = judge_from_verdict(metric, extract_response=extract)

        t = AgentTrace(nodes=[
            TraceNode(name="classify", input="q", output="billing"),
            TraceNode(name="answer", input="q", output="custom response"),
        ])
        judge(t)
        assert metric.calls[0]["response"] == "custom response"

    def test_custom_extract_messages(self):
        metric = _FakeMetric(score=1.0)

        def extract_msgs(trace: AgentTrace) -> list[dict[str, str]]:
            return [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "test"},
            ]

        judge = judge_from_verdict(metric, extract_messages=extract_msgs)
        judge(_make_trace())

        assert metric.calls[0]["messages"] == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "test"},
        ]

    def test_custom_extract_messages_returning_none(self):
        metric = _FakeMetric(score=1.0)
        judge = judge_from_verdict(metric, extract_messages=lambda t: None)
        judge(_make_trace())
        assert metric.calls[0]["messages"] is None


# ---------------------------------------------------------------------------
# Real Verdict integration — keep live while aevyra_verdict is installed.
# ---------------------------------------------------------------------------


verdict = pytest.importorskip("aevyra_verdict")


class TestVerdictIntegration:
    def test_exact_match_correct_response_scores_one(self):
        from aevyra_verdict import ExactMatch
        judge = judge_from_verdict(ExactMatch())
        t = _make_trace(input_="2+2?", output="four", ideal="four")
        assert judge(t) == 1.0

    def test_exact_match_wrong_response_scores_zero(self):
        from aevyra_verdict import ExactMatch
        judge = judge_from_verdict(ExactMatch())
        t = _make_trace(input_="2+2?", output="four", ideal="five")
        assert judge(t) == 0.0

    def test_exact_match_with_non_string_output_uses_json(self):
        from aevyra_verdict import ExactMatch
        judge = judge_from_verdict(ExactMatch())
        # Default extractor json-encodes the output; ideal must match that form.
        t = _make_trace(output={"a": 1}, ideal='{"a": 1}')
        assert judge(t) == 1.0
