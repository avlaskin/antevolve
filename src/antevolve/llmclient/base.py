from abc import ABC, abstractmethod
from typing import Optional, Tuple

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self, 
        prompt: str,
        *,
        model_name: str,
        reasoning: Optional[str] = None,
        max_tokens: int = 40000,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[Optional[str], int, int]:
        """
        Generates text from the LLM.

        Returns:
            Tuple[Optional[str], int, int]: (generated_content, completion_tokens, prompt_tokens)
        """
        pass
