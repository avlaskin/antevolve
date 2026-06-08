from enum import StrEnum

class OperationState(StrEnum):
    """State machine states of the operation."""
    UNDEFINED = 'undefined'
    IN_PROGRESS = 'in_progress'
    STARTED = 'started'
    IN_MUTATION = 'in_mutation'
    IN_EVALUATION = 'in_evaluation'
    FAILED = 'failed'
    FINISHED = 'finished'

class ModelSet(StrEnum):
    """Available model sets configuration."""
    AWS = 'aws'
    LLAMA = 'llama'
    GEMINI = 'gemini'
    MIX = 'mix'
    AWS_MIX = 'aws_mix'
    OPEN_ROUTER = 'open_router'
    OR_QWEN = 'or_qwen'
    OR_ANTHROPIC = 'or_anthropic'
    SINGLE = 'single'
    BEDROCK = 'bedrock'
    BEDMIX = 'bedmix'
    LOCAL = 'local'
    TEST = 'test'


class LLMClientType(StrEnum):
    """Available model clients."""
    OPENAI = 'openai'
    BEDROCK = 'bedrock'
    VERTEXAI = 'vertexai'
