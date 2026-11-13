"""agent-skill-improver: a safe, review-gated daily self-improvement loop for agent skills.

Design rule that governs the whole package: automation may *observe, score,
isolate, propose and validate*. It may never mutate a live skill and it may
never promote its own rubric. Promotion is separate, with review interlocks,
not caller authentication.
"""

__version__ = "0.1.0"

SCHEMA_CONVERSATION = "asi.conversation.v1"
SCHEMA_SCORECARD = "asi.scorecard.v1"
SCHEMA_PROPOSAL = "asi.proposal.v1"
SCHEMA_CANDIDATE = "asi.candidate.v1"
SCHEMA_JUDGEMENT = "asi.judgement.v1"
SCHEMA_RECEIPT = "asi.receipt.v1"
SCHEMA_MODEL_REQUEST = "asi.model_request.v1"
#: The validation record written when a run withheld failures from the improver's prompt
#: (``[caps].unseen_withhold`` > 0). Without a split the writer still writes v3.
SCHEMA_VALIDATION = "asi.validation.v4"
SCHEMA_VALIDATION_V3 = "asi.validation.v3"

__all__ = [
    "__version__",
    "SCHEMA_CONVERSATION",
    "SCHEMA_SCORECARD",
    "SCHEMA_PROPOSAL",
    "SCHEMA_CANDIDATE",
    "SCHEMA_JUDGEMENT",
    "SCHEMA_RECEIPT",
    "SCHEMA_MODEL_REQUEST",
    "SCHEMA_VALIDATION",
    "SCHEMA_VALIDATION_V3",
]
