"""Workflow registry — the generic extension point (A2/R2).

A workflow is a ``WorkflowDefinition`` entry in ``WORKFLOW_REGISTRY``: everything
the runtime needs to validate input, compile, and run it. Introducing a new
workflow is its own subpackage plus one ``register()`` call from that
subpackage's ``__init__`` — no existing graph, the registry, or the runtime
changes (R2). Graph subpackages self-register on import; ``app.graphs``
imports them as they land (T4–T7 each add their import line).

Registry style mirrors ``infrastructure/embedding/factory.py``: a dict keyed
by name, with unknown-name lookups failing fast and listing the known
entries.
"""

import logging
from dataclasses import dataclass
from typing import Callable

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowDefinition:
    """One workflow's registration entry (the ARCH extension-point contract).

    Fields:
        name: unique registry key the runtime facade is invoked with.
        description: human-facing summary (registration log, future listing).
        factory: zero-argument builder of the compiled graph — the runtime's
            compile cache calls it at most once per process.
        input_schema: pydantic model every invocation payload is validated
            against *before* the graph runs.
    """

    name: str
    description: str
    factory: Callable[[], CompiledStateGraph]
    input_schema: type[BaseModel]


# The registry itself — the whole genericity mechanism is this dict plus
# ``register()`` (A2). Populated by graph subpackage imports at startup.
WORKFLOW_REGISTRY: dict[str, WorkflowDefinition] = {}


def register(definition: WorkflowDefinition) -> None:
    """Add ``definition`` to the registry (self-registration hook).

    Re-registering an existing name replaces the entry with a warning —
    importing a graph subpackage more than once must stay harmless.
    """
    if definition.name in WORKFLOW_REGISTRY:
        logger.warning("Workflow %r re-registered; replacing the previous entry",
                       definition.name)
    WORKFLOW_REGISTRY[definition.name] = definition
    logger.info("Workflow registered: %r — %s", definition.name, definition.description)


def get_workflow(name: str) -> WorkflowDefinition:
    """Return the workflow definition registered under ``name``.

    Raises ValueError immediately on an unknown name, listing the known
    workflows (embedding-factory fail-fast style).
    """
    try:
        return WORKFLOW_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(WORKFLOW_REGISTRY)) or "(none registered yet)"
        raise ValueError(
            f"Unknown workflow {name!r}. Known workflows: {known}."
        ) from None
