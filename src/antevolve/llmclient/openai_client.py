import logging
from typing import Optional, Tuple

import openai
from .base import LLMClient

class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible APIs."""

    def __init__(self, api_key: str, base_url: str, max_retries: int = 4, timeout: int = 360):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

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
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort=reasoning,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                cont = response.choices[0].message.content
                comp_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                return cont, comp_tokens, prompt_tokens
            else:
                logging.error(f'Content not found in response: {response}')
                print('Found error: ', response)
                
        except openai.APIError as e:
            logging.error(f"OpenAI API Error: {e}")
            print('Error from OpenAI: ', e)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print('Error from OpenAI: ', e)
            
        return None, 0, 0
