from typing import Any, Optional
from pydantic import BaseModel

from antevolve.models.llmconfig import LLMConfig
from antevolve.models.enums import OperationState

class MutateRequest(BaseModel):
    """Main request that initiates mutation of the program."""
    llm_config: LLMConfig
    program: str
    program_id: str
    evaluator: str
    instruction: str
    scores: Optional[dict[str, Optional[float]]] = None
    max_iterations: int = 1
    timeout: int = 300
    data_path: Optional[str] = None # Either GS/S3 path or local path if run locally.
    eval_llm_config: Optional[LLMConfig] = None


class MutateResponse(BaseModel):
    """Response for the mutate request."""
    operation_id: str
    status: OperationState
    llm_config: LLMConfig
    duration: Optional[float] = None
    start_time: Optional[float] = None
    scores: Optional[dict[str, Optional[float]]] = None
    old_scores: Optional[dict[str, Optional[float]]] = None
    feedback: str | None = None
    final_program: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class StopServiceRequest(BaseModel):
    """Stops all mutation operations."""
    stop_all: bool


class StatusResponse(BaseModel):
    """Returns status."""
    status: bool



