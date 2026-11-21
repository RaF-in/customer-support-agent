"""Conversational orchestrator package (A1) — deliberately NOT registered.

This package is the conversational front door: each turn is classified by
the router into a direct answer, one clarifying question, a single workflow
run, or (slice 2) a planner–executor plan, and every user-facing reply is
composed conversationally from the workflow outcome.

It must never appear in ``WORKFLOW_REGISTRY``: the router/planner menu is
generated from registry contents BY CONSTRUCTION, so registering the
orchestrator would make it routable to itself — and conversely any new
workflow becomes routable with zero orchestrator change (CR-R5/PE-R1).
Nothing imports this package from ``app.graphs``: the conversation service
(``application/conversations.py``) compiles the graph via
``build_orchestrator_graph()`` (exported from T4 on) and invokes it
directly — never through ``run_workflow``.
"""
