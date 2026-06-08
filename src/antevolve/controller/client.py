"""Async client for the Controller Service."""

import httpx
from typing import Optional
from antevolve.models import EvolveResponse
from antevolve.models.llmconfig import LLMConfig

class EvolutionAPIError(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class AsyncEvolutionaryServiceClient:
    """An asynchronous client for the Mutation Service API."""

    def __init__(self, base_url: str, api_key: str):
        """
        Initializes the async client.

        Args:
            base_url: The base URL of the FastAPI service (e.g., "http://127.0.0.1:8000").
            api_key: The API key for authentication.
        """
        self.base_url = base_url.rstrip('/')
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async def _handle_response(self, response: httpx.Response):
        """Checks HTTP response and raises an error if it's not successful."""
        if not response.is_success:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise EvolutionAPIError(status_code=response.status_code, detail=detail)
        return response.json()

    async def start_evolution(
        self,
        program: str,
        evaluator: str,
        instructions: list[str],
        llm_configs: list["LLMConfig"],
        eval_llm_config: Optional["LLMConfig"] = None,
        max_iterations: int = 1,
        count_positive: bool = True,
        operation_id: str | None = None,
        data_path: str | None = None
    ) -> EvolveResponse:
        """
        Starts a new evolution process asynchronously.
        """
        url = f"{self.base_url}/evolve"
        payload = {
            "program": program,
            "evaluator": evaluator,
            "instruction": instructions,
            "llm_configs": [config.model_dump() for config in llm_configs],
            "max_iterations": max_iterations,
            "count_positive": count_positive,
            "data_path": data_path,
        }
        if eval_llm_config:
            payload["eval_llm_config"] = eval_llm_config.model_dump()

        if operation_id:
            payload["operation_id"] = operation_id

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self._headers)
        data = await self._handle_response(response)
        return EvolveResponse(**data)

    async def get_evolution_status(self, operation_id: str) -> EvolveResponse:
        """
        Gets the status of a specific evolution process asynchronously.
        """
        url = f"{self.base_url}/evolve/{operation_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers)
        data = await self._handle_response(response)
        return EvolveResponse(**data)
        
    async def stop_evolution(self, operation_id: str) -> EvolveResponse:
        """
        Stops the process asynchronously.
        """
        url = f"{self.base_url}/evolve/{operation_id}"
        async with httpx.AsyncClient() as client:
            response = await client.delete(url, headers=self._headers)
        data = await self._handle_response(response)
        return EvolveResponse(**data)

    async def upload_data(self, 
        file_path: str | None = None, 
        file_content: bytes | None = None, 
        filename: str | None = None) -> dict:
        """
        Uploads a file to the controller.
        Args:
            file_path: Path to the file to upload.
            file_content: content of the file to uploaded
            filename: name of the file
        """
        url = f"{self.base_url}/upload_data"
        
        # Prepare the file for upload
        import os
        if file_path:
             filename = os.path.basename(file_path)
        
        async with httpx.AsyncClient() as client:
            # We don't set Content-Type header here as httpx handles multipart boundaries
            # We copy headers but exclude Content-Type
            headers = {k: v for k, v in self._headers.items() if k.lower() != 'content-type'}
            headers["X-API-Key"] = self._headers["X-API-Key"]

            if file_path: 
                with open(file_path, "rb") as f:
                    files = {"file": (filename, f, "application/zip")}
                    response = await client.post(url, headers=headers, files=files)
            else:
                 files = {"file": (filename, file_content, "application/zip")}
                 response = await client.post(url, headers=headers, files=files)
                
        if not response.is_success:
             try:
                 detail = response.json().get("detail", response.text)
             except Exception:
                 detail = response.text
             raise EvolutionAPIError(status_code=response.status_code, detail=detail)
        return response.json()
