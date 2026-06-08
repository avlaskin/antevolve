"""Different models config."""
import logging
import os
import random
import numpy as np
import enum
from dataclasses import dataclass
from typing import List
from antevolve.models.enums import ModelSet
from antevolve.models.llmconfig import LLMConfig



def get_models_list(llm_configs: List[LLMConfig], num_models: int = 20000) -> List[LLMConfig]:
    """Generates models list based on probabilities."""
    if not llm_configs:
         return []
         
    # Filter out models with low probability
    llm_configs = [config for config in llm_configs if config.probability >= 0.001]
    
    if not llm_configs:
         return []

    probabilities = [config.probability for config in llm_configs]
    total_probability = sum(probabilities)
    
    if total_probability == 0:
        # Avoid division by zero, equal distribution
        normalized_probabilities = [1.0 / len(probabilities)] * len(probabilities)
    else:
        normalized_probabilities = [p / total_probability for p in probabilities]
    
    for i, config in enumerate(llm_configs):
        config.probability = normalized_probabilities[i]

    # Sample models
    sampled_models = np.random.choice(llm_configs, size=num_models, p=normalized_probabilities)
    return list(sampled_models)
