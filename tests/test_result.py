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

"""Tests for result.py — NodeAttribution, Attribution, PromptAttribution."""

from __future__ import annotations

import json

import pytest

from aevyra_origin.result import (
    Attribution,
    NodeAttribution,
    PromptAttribution,
    VALID_SEVERITIES,
)


# ---------------------------------------------------------------------------
# NodeAttribution
# ---------------------------------------------------------------------------


class TestNodeAttribution:
    def test_construction_minimal(self):
        na = NodeAttribution(
            node_name="classify",
            severity="primary",
            confidence=0.9,
            reasoning="It failed.",
        )
        assert na.node_name == "classify"
        assert na.severity == "primary"
        assert na.confidence == 0.9
        assert na.node_id is None
        assert na.prompt_id is None

    def test_construction_full(self):
        na = NodeAttribution(
            node_name="plan",
            severity="contributing",
            confidence=0.6,
            reasoning="Caused downstream errors.",
            node_id="p1",
            prompt_id="planner",
        )
        assert na.node_id == "p1"
        assert na.prompt_id == "planner"

    def test_bad_severity_raises(self):
        with pytest.raises(ValueError, match="severity must be one of"):
            NodeAttribution(node_name="x", severity="catastrophic", confidence=0.5, reasoning="")

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            NodeAttribution(node_name="x", severity="minor", confidence=-0.1, reasoning="")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            NodeAttribution(node_name="x", severity="minor", confidence=1.1, reasoning="")

    def test_confidence_exactly_zero_and_one_ok(self):
        NodeAttribution(node_name="x", severity="minor", confidence=0.0, reasoning="")
        NodeAttribution(node_name="x", severity="minor", confidence=1.0, reasoning="")

    def test_all_valid_severities(self):
        for sev in VALID_SEVERITIES:
            na = NodeAttribution(node_name="x", severity=sev, confidence=0.5, reasoning="")
            assert na.severity == sev

    def test_to_dict_shape(self):
        na = NodeAttribution(
            node_name="answer",
            severity="primary",
            confidence=0.85,
            reasoning="Bad output.",
            node_id="n2",
            prompt_id="responder",
        )
        d = na.to_dict()
        assert d == {
            "node_name": "answer",
            "severity": "primary",
            "confidence": 0.85,
            "reasoning": "Bad output.",
            "node_id": "n2",
            "prompt_id": "responder",
        }

    def test_from_dict_round_trip(self):
        original = NodeAttribution(
            node_name="retrieve",
            severity="contributing",
            confidence=0.4,
            reasoning="Missed docs.",
            node_id="n1",
            prompt_id="retriever",
        )
        restored = NodeAttribution.from_dict(original.to_dict())
        assert restored.node_name == original.node_name
        assert restored.severity == original.severity
        assert restored.confidence == original.confidence
        assert restored.reasoning == original.reasoning
        assert restored.node_id == original.node_id
        assert restored.prompt_id == original.prompt_id

    def test_from_dict_missing_optional_fields(self):
        d = {"node_name": "foo", "severity": "minor", "confidence": 0.3, "reasoning": "x"}
        na = NodeAttribution.from_dict(d)
        assert na.node_id is None
        assert na.prompt_id is None

    def test_from_dict_confidence_coerced_to_float(self):
        d = {"node_name": "x", "severity": "minor", "confidence": "0.7", "reasoning": ""}
        na = NodeAttribution.from_dict(d)
        assert isinstance(na.confidence, float)
        assert na.confidence == 0.7


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def _make_attribution(**kwargs) -> Attribution:
    defaults = dict(
        summary="Test summary.",
        culprits=[
            NodeAttribution(node_name="a", severity="primary", confidence=0.9, reasoning="A."),
            NodeAttribution(node_name="b", severity="contributing", confidence=0.5, reasoning="B."),
        ],
        method="critic",
        score=0.3,
    )
    defaults.update(kwargs)
    return Attribution(**defaults)


class TestAttribution:
    def test_top_culprit_returns_first(self):
        attr = _make_attribution()
        top = attr.top_culprit()
        assert top is not None
        assert top.node_name == "a"

    def test_top_culprit_empty(self):
        attr = _make_attribution(culprits=[])
        assert attr.top_culprit() is None

    def test_primary_culprits(self):
        attr = _make_attribution()
        primaries = attr.primary_culprits()
        assert len(primaries) == 1
        assert primaries[0].node_name == "a"

    def test_primary_culprits_empty_when_none(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(node_name="x", severity="minor", confidence=0.2, reasoning=""),
            ]
        )
        assert attr.primary_culprits() == []

    def test_to_dict_shape(self):
        attr = _make_attribution()
        d = attr.to_dict()
        assert set(d.keys()) == {"summary", "culprits", "method", "score", "raw", "llm_tokens", "ablation_calls"}
        assert d["method"] == "critic"
        assert d["score"] == 0.3
        assert len(d["culprits"]) == 2

    def test_to_json_is_valid_json(self):
        attr = _make_attribution()
        s = attr.to_json()
        parsed = json.loads(s)
        assert parsed["method"] == "critic"

    def test_to_json_indent(self):
        attr = _make_attribution()
        s = attr.to_json(indent=2)
        assert "\n" in s

    def test_from_dict_round_trip(self):
        original = _make_attribution()
        restored = Attribution.from_dict(original.to_dict())
        assert restored.summary == original.summary
        assert restored.method == original.method
        assert len(restored.culprits) == 2

    def test_from_dict_score_coerced_to_float(self):
        d = {
            "summary": "",
            "culprits": [],
            "method": "critic",
            "score": "0.5",
            "raw": {},
        }
        attr = Attribution.from_dict(d)
        assert isinstance(attr.score, float)

    def test_render_with_culprits(self):
        attr = _make_attribution()
        rendered = attr.render()
        assert "method=critic" in rendered
        assert "score=0.300" in rendered
        assert "primary" in rendered
        assert "a" in rendered

    def test_render_no_culprits(self):
        attr = _make_attribution(culprits=[])
        rendered = attr.render()
        assert "no culprits identified" in rendered

    def test_render_with_node_id(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(
                    node_name="plan",
                    severity="primary",
                    confidence=0.8,
                    reasoning="Planned badly.",
                    node_id="p1",
                )
            ]
        )
        rendered = attr.render()
        assert "id=p1" in rendered


# ---------------------------------------------------------------------------
# Attribution.by_prompt
# ---------------------------------------------------------------------------


class TestByPrompt:
    def test_no_prompt_ids_returns_empty(self):
        attr = _make_attribution()  # default culprits have no prompt_id
        result = attr.by_prompt()
        assert result == []

    def test_single_span_with_prompt_id(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(
                    node_name="plan",
                    severity="primary",
                    confidence=0.8,
                    reasoning="Planning failed.",
                    node_id="p1",
                    prompt_id="planner",
                )
            ]
        )
        result = attr.by_prompt()
        assert len(result) == 1
        pa = result[0]
        assert pa.prompt_id == "planner"
        assert pa.severity == "primary"
        assert abs(pa.confidence - 0.8) < 1e-9
        assert len(pa.spans) == 1

    def test_multiple_spans_same_prompt_id(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(
                    node_name="plan",
                    severity="primary",
                    confidence=0.8,
                    reasoning="Step 1 fail.",
                    node_id="p1",
                    prompt_id="planner",
                ),
                NodeAttribution(
                    node_name="plan",
                    severity="contributing",
                    confidence=0.4,
                    reasoning="Step 2 fail.",
                    node_id="p2",
                    prompt_id="planner",
                ),
            ]
        )
        result = attr.by_prompt()
        assert len(result) == 1
        pa = result[0]
        # mean confidence
        assert abs(pa.confidence - 0.6) < 1e-9
        # max severity
        assert pa.severity == "primary"
        assert len(pa.spans) == 2
        # reasoning contains both spans
        assert "p1" in pa.reasoning or "plan" in pa.reasoning

    def test_spans_without_prompt_id_are_skipped(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(
                    node_name="search",
                    severity="minor",
                    confidence=0.2,
                    reasoning="Tool call failed.",
                    node_id="t1",
                    prompt_id=None,
                ),
                NodeAttribution(
                    node_name="plan",
                    severity="primary",
                    confidence=0.9,
                    reasoning="Planning failed.",
                    node_id="p1",
                    prompt_id="planner",
                ),
            ]
        )
        result = attr.by_prompt()
        assert len(result) == 1
        assert result[0].prompt_id == "planner"

    def test_sorted_by_confidence_descending(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(
                    node_name="a",
                    severity="minor",
                    confidence=0.3,
                    reasoning="A.",
                    node_id="a1",
                    prompt_id="prompt_a",
                ),
                NodeAttribution(
                    node_name="b",
                    severity="primary",
                    confidence=0.9,
                    reasoning="B.",
                    node_id="b1",
                    prompt_id="prompt_b",
                ),
            ]
        )
        result = attr.by_prompt()
        assert result[0].confidence > result[1].confidence

    def test_reasoning_labeled_per_span(self):
        attr = _make_attribution(
            culprits=[
                NodeAttribution(
                    node_name="plan",
                    severity="primary",
                    confidence=0.8,
                    reasoning="Step 1 reasoning.",
                    node_id="p1",
                    prompt_id="planner",
                ),
                NodeAttribution(
                    node_name="plan",
                    severity="contributing",
                    confidence=0.4,
                    reasoning="Step 2 reasoning.",
                    node_id="p2",
                    prompt_id="planner",
                ),
            ]
        )
        result = attr.by_prompt()
        assert len(result) == 1
        reasoning = result[0].reasoning
        assert "p1" in reasoning
        assert "p2" in reasoning
        assert "Step 1 reasoning." in reasoning
        assert "Step 2 reasoning." in reasoning


# ---------------------------------------------------------------------------
# PromptAttribution
# ---------------------------------------------------------------------------


class TestPromptAttribution:
    def test_to_dict_shape(self):
        pa = PromptAttribution(
            prompt_id="planner",
            severity="primary",
            confidence=0.75,
            spans=[
                NodeAttribution(
                    node_name="plan",
                    severity="primary",
                    confidence=0.75,
                    reasoning="Bad plan.",
                    node_id="p1",
                )
            ],
            reasoning="[p1] Bad plan.",
        )
        d = pa.to_dict()
        assert d["prompt_id"] == "planner"
        assert d["severity"] == "primary"
        assert d["confidence"] == 0.75
        assert isinstance(d["spans"], list)
        assert len(d["spans"]) == 1
        assert d["reasoning"] == "[p1] Bad plan."
