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

"""Shared fixtures and helpers for the Origin test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the local witness and origin packages are importable when running
# tests directly from the repo (without a global install).
_repo_root = Path(__file__).parent.parent.parent  # github/
_witness_src = _repo_root / "witness" / "src"
_origin_src = _repo_root / "origin" / "src"
for _p in (_witness_src, _origin_src):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
