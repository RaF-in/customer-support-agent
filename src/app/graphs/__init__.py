"""LangGraph workflow subsystem.

Package layout (as tasks land): ``base/`` holds the shared state envelope;
``registry.py`` + ``runtime.py`` (T3) are the generic extension point and
the single facade every workflow runs through; each workflow is its own
subpackage (``knowledge/`` T4, ``live_data/`` T5, ``actions/`` T6,
``approval/`` T7) that self-registers on import. Graphs never import each
other — cross-graph wiring lives only in the runtime (ARCH Module
Boundaries). This package currently wires only the T1 foundations.
"""

from app.graphs.base.state import WorkflowResult, WorkflowState, WorkflowStatus

__all__ = ["WorkflowResult", "WorkflowState", "WorkflowStatus"]
