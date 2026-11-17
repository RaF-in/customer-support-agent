"""Action contracts — the generic execution surface (T6/R8).

``ActionHandler`` is the protocol every business action implements;
``ActionDefinition`` pairs a name with its input schema; ``ACTION_REGISTRY``
maps names to ``(definition, handler)`` pairs. The registry is the same
dict-plus-register pattern as ``WORKFLOW_REGISTRY`` and the embedding
factory: unknown names fail fast listing the known ones.

Business rules live ONLY in handler implementations (R8) — graph nodes
sequence, persist, and terminate; they never decide eligibility or
thresholds. Mock e-commerce handlers ship first
(``handlers/ecommerce.py``); banking/hospital handlers drop in beside them
without touching the graph, the registry, or the runtime (the R8 proof).
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from uuid import UUID

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ActionContext(BaseModel):
    """What every handler call receives.

    Identity comes from the runtime-injected envelope and nowhere else
    (structural isolation, R3) — a handler cannot see any tenant but its
    caller's. ``approval_id`` is set only when an approved gate re-invokes
    the action (T7 chaining); it is ``None`` on ungated runs.
    """

    tenant_id: str
    run_id: str
    approval_id: Optional[UUID] = None


class ValidationOutcome(BaseModel):
    """Return type of ``ActionHandler.validate`` — business eligibility as a
    value, never an exception (two failure classes, R9)."""

    eligible: bool
    reason: Optional[str] = None  # the handler's reason, present when not eligible


class ActionHandler(Protocol):
    """One business action's adapter (ARCH contract).

    ``inputs`` is the action's validated ``input_schema`` instance (each
    handler is authored beside its own schema, so the protocol stays
    schema-agnostic); ``result`` is the dict ``execute`` returned.
    """

    async def validate(
        self, inputs: Any, context: ActionContext
    ) -> ValidationOutcome:
        """May this action run for these inputs? Business eligibility only —
        schema validity was already settled against ``input_schema`` before
        the graph ran."""
        ...

    async def execute(self, inputs: Any, context: ActionContext) -> dict[str, Any]:
        """Perform the action and return its result payload. Raising here is
        an infrastructure-class failure (runtime → ``failed``), never a
        business outcome."""
        ...

    async def verify(self, result: dict[str, Any], context: ActionContext) -> bool:
        """Confirm the executed result after the fact (reconciliation)."""
        ...


@dataclass(frozen=True)
class ActionDefinition:
    """One action's registration entry — mirrors ``WorkflowDefinition``
    (``graphs/registry.py``): name, human summary, and the pydantic schema
    every input payload is validated against *before* the graph runs."""

    name: str
    description: str
    input_schema: type[BaseModel]


# The registry itself — populated by handler-package imports at startup
# (actions self-register exactly like workflow subpackages).
ACTION_REGISTRY: dict[str, tuple[ActionDefinition, ActionHandler]] = {}


def register_action(definition: ActionDefinition, handler: ActionHandler) -> None:
    """Add one ``(definition, handler)`` pair to the registry.

    Re-registering an existing name replaces the entry with a warning —
    importing a handler module more than once must stay harmless (the same
    stance as ``registry.register``).
    """
    if definition.name in ACTION_REGISTRY:
        logger.warning(
            "Action %r re-registered; replacing the previous entry", definition.name
        )
    ACTION_REGISTRY[definition.name] = (definition, handler)
    logger.info("Action registered: %r — %s", definition.name, definition.description)


def get_action(name: str) -> tuple[ActionDefinition, ActionHandler]:
    """Return the ``(definition, handler)`` pair registered under ``name``.

    Raises ValueError immediately on an unknown name, listing the known
    actions — the fail-fast style shared with ``get_workflow`` and
    ``get_embedding_provider``.
    """
    try:
        return ACTION_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(ACTION_REGISTRY)) or "(none registered yet)"
        raise ValueError(
            f"Unknown action {name!r}. Known actions: {known}."
        ) from None
