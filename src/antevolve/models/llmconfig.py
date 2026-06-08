from pydantic import BaseModel
from enum import StrEnum


class ClientType(StrEnum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    GEMINI = "gemini"


class LLMConfig(BaseModel):
    model_name: str
    base_url: str
    api_key: str
    temperature: float = 1.0
    max_tokens: int = 64000
    llm_client: ClientType = ClientType.OPENAI
    reasoning: str | None = None
    probability: float = 0.01
