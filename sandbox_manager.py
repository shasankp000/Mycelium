from __future__ import annotations

from datetime import datetime

from sandbox_models import SandboxResult, SandboxTask


class SandboxManager:
    """Coordinator for sandbox research runs.

    In this initial PoC implementation, the sandbox is a stub that
    records an empty result. Tool-calling behaviour will be layered
    in on top of this interface.
    """

    def run(self, task: SandboxTask) -> SandboxResult:
        started = datetime.utcnow()
        finished = started
        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=finished,
            steps=[],
            summary=(
                "Sandbox not yet enabled for this trace; this is a stub "
                "result placeholder."
            ),
        )
