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

"""aevyra-origin — failure attribution for agent pipelines.

Three on-ramps, one attribution engine:

1. Turnkey — give Origin your pipeline and it handles tracing + scoring::

       from aevyra_origin import diagnose_pipeline

       result = diagnose_pipeline(
           my_agent, "how do I refund?",
           judge=my_judge, rubric="Accurate and concise.", llm=anthropic_llm(),
       )

2. Adapter — parse framework logs (OpenClaw JSONL today) into an
   ``AgentTrace`` and hand it to Origin::

       from aevyra_witness.adapters import from_openclaw_jsonl
       trace = from_openclaw_jsonl(lines)
       origin.diagnose(trace=trace, score=0.4, rubric=...)

3. Raw — you already have an ``AgentTrace`` and a score::

       from aevyra_origin import Origin
       origin = Origin(llm=anthropic_llm())
       origin.diagnose(trace=my_trace, score=0.4, rubric=...)

For ablation (the causal second-opinion method), construct
``Origin`` with ``runner=`` and ``judge=``, or pass ``runner=`` to
``diagnose_pipeline``.
"""

from aevyra_origin.ablation import AblationError, Judge, Runner, VALID_PLACEHOLDERS
from aevyra_origin.critic import CriticError
from aevyra_origin.decomposition import DecompositionError
from aevyra_origin.diagnose import Origin, VALID_METHODS, diagnose
from aevyra_origin.pipeline import Pipeline, PipelineError, diagnose_pipeline
from aevyra_origin.result import (
    Attribution,
    NodeAttribution,
    PromptAttribution,
    VALID_SEVERITIES,
)

__version__ = "0.1.0"

__all__ = [
    "AblationError",
    "Attribution",
    "CriticError",
    "DecompositionError",
    "Judge",
    "NodeAttribution",
    "Origin",
    "Pipeline",
    "PipelineError",
    "PromptAttribution",
    "Runner",
    "VALID_METHODS",
    "VALID_PLACEHOLDERS",
    "VALID_SEVERITIES",
    "diagnose",
    "diagnose_pipeline",
]
