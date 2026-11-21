"""Deterministic router/planner menu (A1, CR-R7, PE-R1).

``build_catalog()`` renders ``WORKFLOW_REGISTRY`` (plus ``ACTION_REGISTRY``
for ``action_execution`` inputs) into stable menu text with the routing
rules embedded. The menu is registry contents BY CONSTRUCTION: a new
workflow or action becomes routable with zero orchestrator change, and the
orchestrator itself — absent from the registry — can never appear on its
own menu (no self-routing).

Callers must import ``app.graphs`` first: registrations are import-side
effects, and this module deliberately imports only the registries (never a
graph subpackage) per Module Boundaries.
"""

import json

from pydantic import BaseModel

from app.graphs.actions.contracts import ACTION_REGISTRY
from app.graphs.registry import WORKFLOW_REGISTRY

ROUTING_RULES = """\
Routing rules (hard requirements):
- "direct" is reserved for small talk, greetings, and meta questions about
  this assistant. Never use it for questions about products, policies,
  billing, or procedures.
- Knowledge questions — anything about products, policies, billing, or
  procedures that the knowledge base may answer — always route to
  `self_rag`; never answer them directly.
- A compound request that needs more than one workflow must be classified
  `multi_workflow` — never force-fit it into a single workflow.
- Choose `clarify` only when the request is genuinely ambiguous and one
  short question would resolve it; otherwise commit to an outcome."""


def _schema_text(schema: type[BaseModel]) -> str:
    """Render one input JSON schema deterministically."""
    return json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)


def _action_menu() -> str:
    """The per-action menu feeding ``action_execution`` inputs."""
    if not ACTION_REGISTRY:
        return "  (no actions registered yet)"
    lines = [
        "  Available actions (for action_execution's action_name / inputs):"
    ]
    for name in sorted(ACTION_REGISTRY):
        definition, _handler = ACTION_REGISTRY[name]
        lines.append(f"  - {name}: {definition.description}")
        lines.append(f"    input JSON schema: {_schema_text(definition.input_schema)}")
    return "\n".join(lines)


def build_catalog() -> str:
    """Build the full menu text: routing rules + every registered workflow
    (name, description, input JSON schema), with the action menu embedded
    in the ``action_execution`` entry."""
    sections: list[str] = [ROUTING_RULES, "", "Available workflows:"]
    for name in sorted(WORKFLOW_REGISTRY):
        definition = WORKFLOW_REGISTRY[name]
        sections.append(f"- {name}: {definition.description}")
        sections.append(f"  input JSON schema: {_schema_text(definition.input_schema)}")
        if name == "action_execution":
            sections.append(_action_menu())
    return "\n".join(sections)
