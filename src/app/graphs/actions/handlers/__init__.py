"""Mock action handlers — self-register on import (T6/R8).

Importing this package registers every shipped mock action into
``ACTION_REGISTRY``, mirroring graph-subpackage self-registration. A new
handler module follows the same one-line pattern: define schemas + handler,
then ``register_action(ActionDefinition(...), Handler())`` here.
"""

from app.graphs.actions.contracts import ActionDefinition, register_action
from app.graphs.actions.handlers.ecommerce import (
    CancelOrderHandler,
    CancelOrderInput,
    ProcessRefundHandler,
    RefundInput,
)

register_action(
    ActionDefinition(
        name="process_refund",
        description="Refund a customer order (mock e-commerce store)",
        input_schema=RefundInput,
    ),
    ProcessRefundHandler(),
)
register_action(
    ActionDefinition(
        name="cancel_order",
        description="Cancel a customer order (mock e-commerce store)",
        input_schema=CancelOrderInput,
    ),
    CancelOrderHandler(),
)
