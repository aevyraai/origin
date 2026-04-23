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

"""Tests for _json.py — extract_json tolerant parsing."""

from __future__ import annotations

import json

import pytest

from aevyra_origin._json import JSONParseError, extract_json


class TestExtractJson:
    def test_raw_json(self):
        obj = {"key": "value", "num": 42}
        result = extract_json(json.dumps(obj))
        assert result == obj

    def test_raw_json_with_whitespace(self):
        result = extract_json('  {"a": 1}  ')
        assert result == {"a": 1}

    def test_fenced_json_with_json_label(self):
        text = '```json\n{"summary": "ok", "culprits": []}\n```'
        result = extract_json(text)
        assert result["summary"] == "ok"

    def test_fenced_json_without_label(self):
        text = '```\n{"foo": "bar"}\n```'
        result = extract_json(text)
        assert result == {"foo": "bar"}

    def test_json_with_leading_prose(self):
        text = 'Here is my analysis:\n\n{"result": "yes"}'
        result = extract_json(text)
        assert result["result"] == "yes"

    def test_json_with_trailing_prose(self):
        text = '{"result": "yes"}\n\nHope this helps!'
        result = extract_json(text)
        assert result["result"] == "yes"

    def test_json_with_leading_and_trailing_prose(self):
        text = "Sure! Here:\n\n{\"x\": 1}\n\nLet me know if you have questions."
        result = extract_json(text)
        assert result["x"] == 1

    def test_braces_inside_string_values(self):
        # This is the case where naïve brace counting fails.
        obj = {"key": "value with {braces} inside", "num": 3}
        result = extract_json(json.dumps(obj))
        assert result == obj

    def test_nested_objects(self):
        obj = {"outer": {"inner": {"deepest": True}}}
        result = extract_json(json.dumps(obj))
        assert result == obj

    def test_malformed_raises_json_parse_error(self):
        with pytest.raises(JSONParseError):
            extract_json("this is just prose with no JSON")

    def test_malformed_partial_raises(self):
        with pytest.raises(JSONParseError):
            extract_json('{"key": "value"')  # unclosed

    def test_non_string_raises(self):
        with pytest.raises(JSONParseError, match="expected string response"):
            extract_json(None)  # type: ignore[arg-type]

    def test_array_at_top_level_falls_through_to_prose_extraction(self):
        # Arrays are not supported — origin prompts always return objects.
        # This should raise JSONParseError since [1,2,3] has no {}.
        with pytest.raises(JSONParseError):
            extract_json("[1, 2, 3]")

    def test_fenced_malformed_falls_through_to_brace_extraction(self):
        # Fence block is malformed, but raw text has a valid object.
        text = "```json\nnot valid\n```\n\n{\"fallback\": true}"
        result = extract_json(text)
        assert result["fallback"] is True

    def test_complex_llm_response(self):
        # Simulates a real-ish LLM response with explanation + JSON.
        text = (
            "Based on my analysis of the trace, here is the structured output:\n\n"
            "```json\n"
            '{"summary": "The answer node failed.", "culprits": [{"node_name": "answer", "severity": "primary"}]}\n'
            "```\n\n"
            "Let me know if you need more detail."
        )
        result = extract_json(text)
        assert result["summary"] == "The answer node failed."
        assert len(result["culprits"]) == 1

    def test_escaped_quotes_in_strings(self):
        obj = {"key": 'He said "hello" and left'}
        result = extract_json(json.dumps(obj))
        assert result["key"] == 'He said "hello" and left'

    def test_backslash_escape_in_string(self):
        obj = {"path": "C:\\Users\\test"}
        result = extract_json(json.dumps(obj))
        assert result["path"] == "C:\\Users\\test"
