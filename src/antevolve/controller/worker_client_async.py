"""Async client for the mutation service using httpx."""
import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from antevolve.filelib.file_ops import read_text_file as load_text_file
from antevolve.models import LLMConfig, MutateRequest, MutateResponse, StatusResponse


class MutationServiceClientAsync:
    """A Python async client for the Mutation Service API using httpx."""

    def __init__(self, base_url: str, api_key: str = "QWERTY12345"):
        """
        Initializes the client.

        Args:
            base_url (str): The base URL of the FastAPI service (e.g., http://127.0.0.1:8000).
            api_key (str): The API key for authentication.
        """
        # Ensure the base_url doesn't have a trailing slash
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        # We will use a context manager creating clients per request or manage a persistent client.
        # For simplicity in async usage often it's better to pass a client or create one.
        # Here we'll create a new client for requests to avoid lifecycle management issues 
        # unless used as a context manager itself. 
        # However, for efficiency, keeping a client is better. 
        # Let's rely on httpx.AsyncClient() context managers in individual requests or methods if we want to be stateless,
        # but for a service class, having a `close` method or using it as a context manager is standard.
        # Given the usage pattern in service.py (creating new client in a loop), 
        # lightweight instantiation is preferred, or proper cleanup.
        # For this implementation, we will use a persistent client but caller should close it, 
        # OR we use the static httpx methods / context managers inside.
        # Let's use context managers inside methods for robust simple usage without explicit close requirements for now,
        # matching the "requests" sync style where session management was implicit/handled by Garbage Collection mostly (though Session should be closed).
        
        # Actually requests.Session() was used in original. 
        # Let's implement an async context manager pattern for the class itself, or just use one-off clients if acceptable.
        # Better: use an instance client and provide a close method.
        
        self.timeout = httpx.Timeout(120.0, connect=5.0)

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=100),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Internal helper function to make requests and handle responses.

        Args:
            method (str): The HTTP method (e.g., 'GET', 'POST', 'DELETE').
            endpoint (str): The API endpoint path (e.g., '/status').
            **kwargs: Additional arguments for the request (e.g., json payload).

        Returns:
            The JSON response from the API.

        Raises:
            httpx.HTTPStatusError: For non-2xx responses.
            httpx.RequestError: For network-related errors.
        """
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
            try:
                response = await client.request(method, url, **kwargs)
                # Raise an exception for bad status codes (4xx or 5xx)
                response.raise_for_status()
                # Handle empty responses for success codes like 204
                return response.json() if response.text else None
            except httpx.HTTPStatusError as e:
                print(f"HTTP Error: {e.response.status_code} for URL: {e.request.url}", file=sys.stderr)
                print(f"Response: {e.response.text}", file=sys.stderr)
                raise
            except httpx.RequestError as e:
                print(f"A network error occurred: {e}", file=sys.stderr)
                raise

    # --- API Methods ---

    async def get_status(self) -> StatusResponse:
        """
        Checks the health status of the service.
        Corresponds to `GET /status`.
        """
        print("ACTION: Checking service health...")
        response_json = await self._request("GET", "/status")
        return StatusResponse.model_validate(response_json)

    async def start_mutation(
        self,
        program: str,
        evaluator: str,
        instruction: str,
        program_id: str,
        base_url: str,
        api_key: str,
        max_new_tokens: int = 40_000,
        scores: dict[str, float] | None = None,
        model_name: str = 'gemini-2.0-flash',
        data_path: str | None = None,
        max_iterations: int = 1,
        temperature: float = 0.5,
        timeout: int = 300,
        eval_llm_config: Optional[LLMConfig] = None,
    ) -> MutateResponse:
        """
        Starts a new mutation process.
        Corresponds to `POST /mutate`.
        """
        llm_config = LLMConfig(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_new_tokens,
            api_key=api_key
        )
        request = MutateRequest(
            llm_config=llm_config,
            program=program,
            program_id=program_id,
            evaluator=evaluator,
            data_path=data_path,
            instruction=instruction,
            max_iterations=max_iterations,
            scores=scores,
            timeout=timeout,
            eval_llm_config=eval_llm_config
        )
        response_json = await self._request("POST", "/mutate", json=request.model_dump())
        return MutateResponse.model_validate(response_json)

    async def list_mutations(self) -> list[MutateResponse]:
        """
        Lists all active mutation processes.
        Corresponds to `GET /mutate`.
        """
        print("ACTION: Listing all active mutations...")
        response_json = await self._request("GET", "/mutate")
        if isinstance(response_json, dict) and "message" in response_json:
             return []
        return [MutateResponse.model_validate(x) for x in response_json]

    async def get_mutation_status(self, operation_id: str) -> MutateResponse:
        """
        Gets the status of a specific mutation process.
        Corresponds to `GET /mutate/{operation_id}`.
        """
        response_json = await self._request("GET", f"/mutate/{operation_id}")
        return MutateResponse.model_validate(response_json)

    async def stop_mutation(self, operation_id: str) -> MutateResponse:
        """
        Stops a specific mutation process.
        Corresponds to `DELETE /mutate/{operation_id}`.
        """
        print(f"ACTION: Stopping operation '{operation_id}'...")
        response_json =  await self._request("DELETE", f"/mutate/{operation_id}")
        return MutateResponse.model_validate(response_json)

    async def stop_all_mutations(self) -> Dict[str, str]:
        """
        Stops all running mutation processes.
        Corresponds to `POST /stop`.
        """
        print("ACTION: Stopping all mutations...")
        payload = {"stop_all": True}
        return await self._request("POST", "/stop", json=payload)

# --- Sample Data ---
sample_program = """
def addition(a: int, b: int) -> int :
    # This is an int eddition function.
    return 0
"""
sample_evaluator = """
from program import addition

def evaluate(data: str):
    # Evaluates the program from top.
    score = 0.0
    if addition(1, 1) == 2:
        score += 0.5
    if addition(3, 5) == 8:
        score += 0.5
    return {'score': score}
"""

async def main():
    """
    Main function to demonstrate the client's functionality.
    """
    # --- Configuration ---
    # Change this if your service is running elsewhere
    BASE_URL = "http://127.0.0.1:9001" 
    API_KEY = "QWERTY12345"  # The default API key from the service

    # --- Dummy Content for the Request ---
    dummy_program = sample_program
    dummy_evaluator = sample_evaluator
    # Try loading files if they exist, otherwise use samples
    try:
        dummy_program = load_text_file('./data/program.py')
    except:
        pass
    try:
        dummy_evaluator = load_text_file('./data/evaluator.py')
    except:
        pass
        
    dummy_instruction = "Change the program to pass evaluation."

    print("--- Mutation Service Async Client Demo ---")
    
    try:
        # Initialize the client
        api_key = os.environ.get("GOOGLE_API_KEY")
        client = MutationServiceClientAsync(base_url=BASE_URL, api_key=API_KEY)

        # 1. Check service health
        status = await client.get_status()
        if not status.get("status"):
            print("❌ Service is down. Exiting.")
            return
        print("✅ Service is up and running.\n")

        # 2. Start a mutation
        start_response = await client.start_mutation(
            program=dummy_program,
            evaluator=dummy_evaluator,
            instruction=dummy_instruction,
            program_id='program_001',
            base_url="http://localhost:8000", # Dummy base_url for the worker
            api_key=api_key
        )
        operation_id = start_response.get("operation_id")
        print(f"✅ Mutation process started with Operation ID: {operation_id}\n")

        # 3. Poll for the status until it's done
        print(f"--- Polling status for operation {operation_id} ---")
        final_status = None
        while True:
            status_response = await client.get_mutation_status(operation_id)
            current_status = status_response.get("status")
            print(f"   Current status: '{current_status}'")

            if current_status in ["finished", "failed"]:
                final_status = status_response
                print(f"✅ Operation completed with final status: '{current_status}'")
                break
            
            await asyncio.sleep(5)  # Wait before polling again

        print("\n--- Final Operation Details ---")
        print(json.dumps(final_status, indent=2))
        print("-" * 33)

    except httpx.RequestError:
        print("\n❌ A network error occurred.", file=sys.stderr)
        print("   Please ensure the FastAPI service is running and accessible.", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
