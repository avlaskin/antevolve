"""Simple client for the mutation service."""
import requests
import time
import os
import sys
import json
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import Dict, Any, Optional
from fileops import load_text_file
from models.llmconfig import LLMConfig

from tenacity import retry, stop_after_attempt, wait_exponential


class MutationServiceClient:
    """A Python client for the Mutation Service API."""

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
        
        # Set up a session to automatically include the API key in all requests
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
        self.session.headers.update({"Content-Type": "application/json"})

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=120))
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Internal helper function to make requests and handle responses.

        Args:
            method (str): The HTTP method (e.g., 'GET', 'POST', 'DELETE').
            endpoint (str): The API endpoint path (e.g., '/status').
            **kwargs: Additional arguments for the request (e.g., json payload).

        Returns:
            The JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: For non-2xx responses.
            requests.exceptions.RequestException: For network-related errors.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=10, **kwargs)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            # Handle empty responses for success codes like 204
            return response.json() if response.text else None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code} for URL: {e.response.url}", file=sys.stderr)
            print(f"Response: {e.response.text}", file=sys.stderr)
            raise
        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}", file=sys.stderr)
            raise

    # --- API Methods ---

    def get_status(self) -> Dict[str, bool]:
        """
        Checks the health status of the service.
        Corresponds to `GET /status`.
        """
        print("ACTION: Checking service health...")
        return self._request("GET", "/status")

    def start_mutation(
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
        data_path: str = "",
        max_iterations: int = 1,
        temperature: float = 0.5,
    ) -> Dict[str, str]:
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
        payload = {
            "llm_config": llm_config.model_dump(),
            "program": program,
            "program_id": program_id,
            "evaluator": evaluator,
            "data_path": data_path,
            "instruction": instruction,
            "max_iterations": max_iterations,
            "scores": scores,
        }
        return self._request("POST", "/mutate", json=payload)

    def list_mutations(self) -> Dict[str, Any]:
        """
        Lists all active mutation processes.
        Corresponds to `GET /mutate`.
        """
        print("ACTION: Listing all active mutations...")
        return self._request("GET", "/mutate")

    def get_mutation_status(self, operation_id: str) -> Dict[str, Any]:
        """
        Gets the status of a specific mutation process.
        Corresponds to `GET /mutate/{operation_id}`.
        """
        return self._request("GET", f"/mutate/{operation_id}")

    def stop_mutation(self, operation_id: str) -> Dict[str, str]:
        """
        Stops a specific mutation process.
        Corresponds to `DELETE /mutate/{operation_id}`.
        """
        print(f"ACTION: Stopping operation '{operation_id}'...")
        return self._request("DELETE", f"/mutate/{operation_id}")

    def stop_all_mutations(self) -> Dict[str, str]:
        """
        Stops all running mutation processes.
        Corresponds to `POST /stop`.
        """
        print("ACTION: Stopping all mutations...")
        payload = {"stop_all": True}
        return self._request("POST", "/stop", json=payload)

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



def main():
    """
    Main function to demonstrate the client's functionality.
    
    This function will:
    1. Check if the service is running.
    2. Start a new mutation job.
    3. Poll the job's status until it is finished or fails.
    4. Display the final result.
    5. Stop the (now completed) job.
    6. Stop all jobs on the server.
    """
    # --- Configuration ---
    # Change this if your service is running elsewhere
    BASE_URL = "http://127.0.0.1:9001" 
    API_KEY = "QWERTY12345"  # The default API key from the service

    # --- Dummy Content for the Request ---
    dummy_program = sample_program
    dummy_evaluator = sample_evaluator
    dummy_program = load_text_file('./data/program.py')
    dummy_evaluator = load_text_file('./data/evaluator.py')
    dummy_instruction = "Change the program to pass evaluation."

    print("--- Mutation Service Client Demo ---")
    
    try:
        # Initialize the client
        client = MutationServiceClient(base_url=BASE_URL, api_key=API_KEY)

        # 1. Check service health
        status = client.get_status()
        if not status.get("status"):
            print("❌ Service is down. Exiting.")
            return
        print("✅ Service is up and running.\n")

        # 2. Start a mutation
        start_response = client.start_mutation(
            program=dummy_program,
            evaluator=dummy_evaluator,
            instruction=dummy_instruction,
            program_id='program_001',
        )
        operation_id = start_response.get("operation_id")
        print(f"✅ Mutation process started with Operation ID: {operation_id}\n")

        # 3. Poll for the status until it's done
        print(f"--- Polling status for operation {operation_id} ---")
        final_status = None
        while True:
            status_response = client.get_mutation_status(operation_id)
            current_status = status_response.get("status")
            print(f"   Current status: '{current_status}'")

            # The server uses StrEnum, which serializes to lowercase strings
            if current_status in ["finished", "failed"]:
                final_status = status_response
                print(f"✅ Operation completed with final status: '{current_status}'")
                break
            
            time.sleep(5)  # Wait before polling again

        print("\n--- Final Operation Details ---")
        print(json.dumps(final_status, indent=2))
        print("-" * 33)

        # 4. Stop the completed operation to clean up
        #stop_response = client.stop_mutation(operation_id)
        #print(f"✅ Cleanup successful: {stop_response.get('message')}\n")

        # 5. Stop all services (for demonstration)
        #stop_all_response = client.stop_all_mutations()
        #print(f"✅ Server-wide stop successful: {stop_all_response.get('message')}")

    except requests.exceptions.RequestException:
        print("\n❌ A network error occurred.", file=sys.stderr)
        print("   Please ensure the FastAPI service is running and accessible.", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
