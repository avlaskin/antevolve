"""Defines key classes for evolve service."""
from pydantic import BaseModel, field_validator
from typing import Optional, List
from antevolve.models.llmconfig import LLMConfig

# --- API Request/Response ---
class EvolveRequest(BaseModel):
    """Main request that initiates mutation of the program."""
    program: str
    evaluator: str
    instruction: List[str]
    utils: Optional[str] = None
    max_iterations: int = 1
    count_positive: bool = True
    llm_configs: List[LLMConfig]
    eval_llm_config: Optional[LLMConfig] = None
    operation_id: Optional[str] = None # set in case of the operation continue
    data_path: Optional[str] = None

    @field_validator('llm_configs')
    @classmethod
    def check_llm_configs(cls, v: List[LLMConfig]) -> List[LLMConfig]:
        if not v:
            raise ValueError('llm_configs cannot be empty')
        
        for config in v:
            if config.probability < 0.001:
                raise ValueError(f'Probability for model {config.model_name} must be >= 0.001')
        return v


class EvolveIterationsInfo(BaseModel):
    max_iterations: int = 0
    done_iterations: int = 0
    total_programs: int = 0


class EvolveResponse(BaseModel):
    """Main request that initiates mutation of the program."""
    operation_id: str
    best_program: str
    iterations_info: EvolveIterationsInfo
    status: str = ''    
