"""MCP tool infrastructure — settings-declared servers + tenant allowlist (T5/R6/R3).

``MCP_SERVERS`` (JSON, langchain-mcp-adapters client format) declares the
tool servers; ``MCP_TENANT_ALLOWLIST`` (JSON, ``tenant -> [tool names]``)
declares which tenant may use which tool. ``load_mcp_tools(tenant_id)`` is
the single door the live-data graph uses: it lists tools from the configured
servers and returns exactly the tenant's allowlisted subset — never the full
set, never an unfiltered tool.

Isolation is structural (R3, the task's high-risk callout): the allowlist
filter runs at *tool-load time* — the tenant sees only its tools before
anything is bound to a model — never as a post-hoc prompt filter. The filter
is deny-by-default (developer decision 4): a tenant absent from the
allowlist (or with an empty entry) is refused outright with a clear reason
instead of receiving an empty toolbox, so a misconfigured tenant fails
loudly at run start rather than silently answering without tools.

Session-lifecycle choice (documented per task): tools load through an
*unconnected* ``MultiServerMCPClient`` — ``get_tools()`` opens a
short-lived session per server to list tools, and every returned tool opens
a fresh session per invocation. Nothing (client, session, stdio subprocess)
is held open across a graph run, so there is nothing to leak when a run
ends in any state; the cost is one session handshake per tool call — the
right trade for a stdio-subprocess topology at MVP scale. A cached
long-lived client would save handshakes but pin subprocesses per process
for zero steady traffic; revisit with the orchestrator phase.
"""

import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


def _normalized_servers(servers: dict[str, Any]) -> dict[str, Any]:
    """Accept the documented transport-less stdio form of ``MCP_SERVERS``.

    ``.env.example`` documents stdio entries as ``{"command": …, "args":
    […]}`` without a ``transport`` key; the adapters client requires the
    key. Entries with ``command`` and no ``transport`` get ``"stdio"``
    injected; anything else passes through untouched and fails inside the
    client with its own clear configuration error.
    """
    normalized: dict[str, Any] = {}
    for name, connection in servers.items():
        if (
            isinstance(connection, dict)
            and "command" in connection
            and "transport" not in connection
        ):
            normalized[name] = {**connection, "transport": "stdio"}
        else:
            normalized[name] = connection
    return normalized


async def load_mcp_tools(
    tenant_id: str, settings: Optional[Settings] = None
) -> list[BaseTool]:
    """Load this tenant's allowlisted MCP tools (deny-by-default).

    The allowlist is checked *before* any server is contacted, so a denied
    tenant costs no session handshakes. Raises — the live-data graph and the
    runtime facade turn the raise into a failed run — rather than returning
    an empty toolbox: an unusable configuration must fail loudly, not
    silently degrade into a tool-less agent.

    Raises:
        ValueError: no MCP servers configured; the tenant is not
            allowlisted; or the tenant's allowlist names tools that do not
            exist on any configured server.
        RuntimeError: a configured server could not be reached or spawned,
            or otherwise failed to list its tools.
    """
    settings = settings or Settings()
    servers = settings.MCP_SERVERS or {}
    if not servers:
        raise ValueError(
            "no MCP servers configured (MCP_SERVERS is empty) — the"
            " live-data workflow has no tools to run with"
        )

    allowed_names = (settings.MCP_TENANT_ALLOWLIST or {}).get(tenant_id) or []
    if not allowed_names:
        raise ValueError(
            f"tenant {tenant_id!r} is not allowlisted for any MCP tools"
            " (MCP_TENANT_ALLOWLIST is deny-by-default: add the tenant with"
            " its tool names, or do not run live-data workflows for it)"
        )

    client = MultiServerMCPClient(connections=_normalized_servers(servers))
    try:
        loaded = await client.get_tools()
    except Exception as error:
        raise RuntimeError(
            f"MCP tool load failed while contacting servers"
            f" {sorted(servers)}: {error!r}"
        ) from error

    allowed = set(allowed_names)
    tools = [tool for tool in loaded if tool.name in allowed]
    if not tools:
        loaded_names = sorted(tool.name for tool in loaded)
        raise ValueError(
            f"MCP allowlist misconfiguration for tenant {tenant_id!r}: none"
            f" of its allowed tools {sorted(allowed)} exist on the"
            f" configured servers (loaded tools: {loaded_names})"
        )
    logger.info(
        "MCP tools ready for tenant %s: %s",
        tenant_id,
        sorted(tool.name for tool in tools),
    )
    return tools
