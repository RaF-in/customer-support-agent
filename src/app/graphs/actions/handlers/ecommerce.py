"""Mock e-commerce action handlers (T6/R8) — the drop-in proof, not shipped
systems.

Business rules live ONLY here, never in graph nodes: eligibility reads a
per-tenant in-memory order store, execution mutates it, verification
re-reads it. The store is process-local and starts empty; ``seed_orders`` /
``reset_store`` exist for run-once verification and demos. Swapping these
mocks for real integrations — or adding banking/hospital handlers beside
them — touches this package alone; the graph, registries, and runtime stay
unchanged (the R8 proof the ARCH's genericity claim rests on).
"""

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from app.graphs.actions.contracts import ActionContext, ValidationOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The mock store — tenant_id -> order_id -> MockOrder (tenant-scoped like
# every other data surface in the system, R3).
# ---------------------------------------------------------------------------


@dataclass
class MockOrder:
    order_id: str
    total: float
    refunded_total: float = 0.0
    status: str = "confirmed"  # confirmed | cancelled (the mock lifecycle)


_ORDERS: dict[str, dict[str, MockOrder]] = {}


def seed_orders(tenant_id: str, orders: list[MockOrder]) -> None:
    """Pre-seed one tenant's orders (verification / demo convenience)."""
    _ORDERS.setdefault(tenant_id, {}).update(
        {order.order_id: order for order in orders}
    )


def get_order(tenant_id: str, order_id: str) -> Optional[MockOrder]:
    """Read one order from the tenant's store (None when unknown)."""
    return _ORDERS.get(tenant_id, {}).get(order_id)


def reset_store() -> None:
    """Drop every tenant's orders (run-once verification hygiene)."""
    _ORDERS.clear()


# ---------------------------------------------------------------------------
# process_refund — the ARCH reference action
# ---------------------------------------------------------------------------


class RefundInput(BaseModel):
    """Schema for ``process_refund`` inputs."""

    order_id: str = Field(min_length=1)
    amount: float = Field(gt=0)
    reason: str = ""


class ProcessRefundHandler:
    """Refund an order in the mock store."""

    async def validate(
        self, inputs: RefundInput, context: ActionContext
    ) -> ValidationOutcome:
        order = get_order(context.tenant_id, inputs.order_id)
        if order is None:
            return ValidationOutcome(
                eligible=False,
                reason=f"order {inputs.order_id} not found for this tenant",
            )
        if order.status == "cancelled":
            return ValidationOutcome(
                eligible=False,
                reason=f"order {inputs.order_id} is cancelled and cannot be refunded",
            )
        remaining = order.total - order.refunded_total
        if inputs.amount > remaining:
            return ValidationOutcome(
                eligible=False,
                reason=(
                    f"refund of {inputs.amount} exceeds the remaining"
                    f" refundable {remaining} on order {inputs.order_id}"
                ),
            )
        return ValidationOutcome(eligible=True)

    async def execute(
        self, inputs: RefundInput, context: ActionContext
    ) -> dict[str, object]:
        order = get_order(context.tenant_id, inputs.order_id)
        order.refunded_total += inputs.amount
        order_status = (
            "refunded" if order.refunded_total >= order.total else "partially_refunded"
        )
        return {
            "order_id": order.order_id,
            "refunded_amount": inputs.amount,
            "refunded_total": order.refunded_total,
            "order_status": order_status,
        }

    async def verify(
        self, result: dict[str, object], context: ActionContext
    ) -> bool:
        order = get_order(context.tenant_id, str(result.get("order_id")))
        return order is not None and order.refunded_total >= float(
            result.get("refunded_total", 0.0)
        )


# ---------------------------------------------------------------------------
# cancel_order — the no-policy-entry action (never gated, decision 5)
# ---------------------------------------------------------------------------


class CancelOrderInput(BaseModel):
    """Schema for ``cancel_order`` inputs."""

    order_id: str = Field(min_length=1)
    note: str = ""


class CancelOrderHandler:
    """Cancel an order in the mock store."""

    async def validate(
        self, inputs: CancelOrderInput, context: ActionContext
    ) -> ValidationOutcome:
        order = get_order(context.tenant_id, inputs.order_id)
        if order is None:
            return ValidationOutcome(
                eligible=False,
                reason=f"order {inputs.order_id} not found for this tenant",
            )
        if order.status == "cancelled":
            return ValidationOutcome(
                eligible=False,
                reason=f"order {inputs.order_id} is already cancelled",
            )
        return ValidationOutcome(eligible=True)

    async def execute(
        self, inputs: CancelOrderInput, context: ActionContext
    ) -> dict[str, object]:
        order = get_order(context.tenant_id, inputs.order_id)
        order.status = "cancelled"
        return {"order_id": order.order_id, "order_status": "cancelled"}

    async def verify(
        self, result: dict[str, object], context: ActionContext
    ) -> bool:
        order = get_order(context.tenant_id, str(result.get("order_id")))
        return order is not None and order.status == "cancelled"
