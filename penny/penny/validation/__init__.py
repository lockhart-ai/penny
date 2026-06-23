"""Penny's model-I/O validation layer.

Three modules:

- ``conditions`` — the single behaviour taxonomy: every condition we classify
  Penny's model behaviour through, defined once.  A dependency-light leaf (only
  ``constants`` + pydantic), imported by the live loop, the post-hoc run-health
  classifier, the addon, and Penny's own ``quality`` self-review.
- ``outcomes`` — the ``ValidationOutcome`` disposition union a
  ``ResponseValidator`` returns, the ``LoopContext`` it reads, and the
  ``run_validators`` dispatcher.
- ``response_validators`` — the concrete validators.

Only ``conditions`` and ``outcomes`` are re-exported here: both are
database-free leaves, so importing this package (or a submodule of it) during
``penny.database`` initialisation is safe.  ``response_validators`` is NOT
re-exported — it imports ``tools.memory_tools`` (and thus ``penny.database``),
so eagerly importing it here would close an import cycle when the database layer
imports the ``conditions`` leaf.  Import it directly:
``from penny.validation.response_validators import ...``.

See ``docs/model-io-validation.md`` for the design.
"""

from penny.validation.conditions import (
    CATALOG,
    BehaviorCondition,
    ConditionKey,
    condition,
    run_flag_conditions,
)
from penny.validation.outcomes import (
    LoopContext,
    NudgeContinue,
    Proceed,
    RejectToolCall,
    Repair,
    ResponseValidator,
    Retry,
    Stop,
    ValidationOutcome,
    run_validators,
)

__all__ = [
    "CATALOG",
    "BehaviorCondition",
    "ConditionKey",
    "LoopContext",
    "NudgeContinue",
    "Proceed",
    "RejectToolCall",
    "Repair",
    "ResponseValidator",
    "Retry",
    "Stop",
    "ValidationOutcome",
    "condition",
    "run_flag_conditions",
    "run_validators",
]
