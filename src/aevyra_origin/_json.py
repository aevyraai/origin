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

"""Tolerant JSON parsing for LLM responses.

LLMs don't always respect "respond with JSON only" — they wrap it in
markdown fences, prepend a sentence, or trail with an explanation. This
module extracts the first valid JSON object from a messy response.
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    """Raised when a response contains no parseable JSON object."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a single JSON object from an LLM response.

    Tries, in order:

    1. The raw text as JSON.
    2. The contents of the first ``` fenced block.
    3. The substring from the first ``{`` to the matching ``}``.

    Args:
        text: Raw LLM response.

    Returns:
        The parsed JSON object (always a dict — arrays and scalars are not
        supported because all Origin prompts return objects).

    Raises:
        JSONParseError: if no valid JSON object can be extracted.
    """
    if not isinstance(text, str):
        raise JSONParseError(f"expected string response, got {type(text).__name__}")

    # 1. Raw
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    match = _FENCE_RE.search(stripped)
    if match:
        body = match.group(1).strip()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. First balanced {...} — handles strings/escapes properly.
    obj = _extract_first_object(stripped)
    if obj is not None:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            raise JSONParseError(
                f"found a JSON-like block but could not parse it: {e}\n"
                f"---\n{obj[:500]}\n---"
            ) from e

    raise JSONParseError(
        "no parseable JSON object found in response\n"
        f"---\n{stripped[:500]}\n---"
    )


def _extract_first_object(s: str) -> str | None:
    """Find the first ``{...}`` balanced object in ``s``.

    String- and escape-aware: braces inside JSON strings do not affect the
    balance count. Returns the substring (inclusive of outer braces) or
    None if no balanced object is found.
    """
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


__all__ = ["JSONParseError", "extract_json"]
