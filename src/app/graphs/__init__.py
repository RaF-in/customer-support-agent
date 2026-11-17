"""LangGraph workflow subsystem.

Package layout (as tasks land): ``base/`` holds the shared state envelope;
``registry.py`` + ``runtime.py`` (T3) are the generic extension point and
the single facade every workflow runs through; each workflow is its own
subpackage (``knowledge/`` T4, ``live_data/`` T5, ``actions/`` T6,
``approval/`` T7) that self-registers on import. Graphs never import each
other — cross-graph wiring lives only in the runtime (ARCH Module
Boundaries). Landed subpackages: ``knowledge/`` (T4), ``live_data/`` (T5),
``actions/`` (T6). The ``approval/`` package exists with its policy engine
only (landed with T6 — the actions gate consumes it); its graph is T7's.
"""

from app.graphs import actions, knowledge, live_data  # noqa: F401  (import = self-registration)
from app.graphs.base.state import WorkflowResult, WorkflowState, WorkflowStatus

__all__ = ["WorkflowResult", "WorkflowState", "WorkflowStatus"]
