"""LangGraph workflow subsystem.

Package layout (as tasks land): ``base/`` holds the shared state envelope;
``registry.py`` + ``runtime.py`` (T3) are the generic extension point and
the single facade every workflow runs through; each workflow is its own
subpackage (``knowledge/`` T4, ``live_data/`` T5, ``actions/`` T6) that
self-registers on import. Graphs never import each other — cross-graph
wiring lives only in the runtime (ARCH Module Boundaries). Landed
subpackages: ``knowledge/`` (T4), ``live_data/`` (T5), ``actions/`` (T6).
The ``approval/`` package holds the policy engine only (the actions gate
consumes it); the interrupt-based approval graph it once had was deleted
with the checkpointer — approvals pause on a Postgres row owned by the
runtime facade.
"""

from app.graphs import actions, knowledge, live_data  # noqa: F401  (import = self-registration)
from app.graphs.base.state import WorkflowResult, WorkflowState, WorkflowStatus

__all__ = ["WorkflowResult", "WorkflowState", "WorkflowStatus"]
