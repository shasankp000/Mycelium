"""
expert_post_check
=================
Expert Post-Check System.

Verifies the output of the selected domain expert (USE_EXISTING_EXPERT path)
before handing the answer to the conversation layer LLM.

Public API
----------
    from expert_post_check import PostCheckRunner, PostCheckResult
    runner = PostCheckRunner()
    result = runner.run(query, expert_answer, domain, expert_name)
"""

from mycelium.pipeline.expert_post_check.runner import PostCheckRunner, PostCheckResult
from mycelium.pipeline.expert_post_check.hardware_check import probe_hardware, PostCheckMode

__all__ = [
    "PostCheckRunner",
    "PostCheckResult",
    "probe_hardware",
    "PostCheckMode",
]
