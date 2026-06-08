import time
from typing import Dict, Optional, Sequence
from pydantic import BaseModel, Field

class Program(BaseModel):
    """
    Represents a program in the evolutionary database.
    """
    program_id: Optional[str] = None
    island_id: Optional[int] = None
    generation: int = 0
    content: Dict[str, str]
    scores: Dict[str, Optional[float]]
    parent_ids: Sequence[str] = Field(default_factory=list)
    created: float = Field(default_factory=time.time)
    metadata: Optional[Dict[str, str|float|int]] = None
