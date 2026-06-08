import asyncio
import sys
import os
import argparse


sys.path.append(os.path.join(os.getcwd(), 'src'))

from antevolve.controller.client import AsyncEvolutionaryServiceClient, EvolutionAPIError
from antevolve.filelib.file_ops import read_text_file
from antevolve.models.llmconfig import LLMConfig, ClientType

_CONTROLLER_URL = "http://localhost:8989"
_USE_GEMINI=True

async def main():
    parser = argparse.ArgumentParser(description="Experiment Zero: Remote Controller Test")
    parser.add_argument("--controller-url", default=_CONTROLLER_URL, help="URL of the remote controller service")
    parser.add_argument("--api-key", default="", help="API Key for the controller")
    parser.add_argument("--data-path", default="/app/data/data.zip", help="Path to the data folder")
    parser.add_argument("--max-turns", default=51, type=int, help="Number of turns")
    
    args = parser.parse_args()

    controller_url = args.controller_url
    api_key = args.api_key
    data_path = args.data_path
    max_turns = args.max_turns

    print(f"--- Starting Experiment Zero ---")
    print(f"Target Controller: {controller_url}")
    print(f"Data Path: {data_path}")

    client = AsyncEvolutionaryServiceClient(base_url=controller_url, api_key=api_key)

    # Sample Data
    program = read_text_file('./experiment/program.py')
    evaluator = read_text_file('./experiment/evaluator.py')
    status_data = None

    if _USE_GEMINI:
        models = [
            'gemini-3-flash-preview',
            'gemini-3.1-pro-preview'
        ]
        llm_api_key = os.getenv("GOOGLE_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/"
        if not llm_api_key:
            print("Error: GOOGLE_API_KEY is not set. Set GOOGLE_API_KEY environment variable: export GOOGLE_API_KEY=your-api-key")
            sys.exit(1)
    else:
        models = [
            'qwen/qwen3-next-80b-a3b-thinking',
            'qwen/qwen3-coder'
        ]
        llm_api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        if not llm_api_key:
            print("Error: OPENROUTER_API_KEY is not set. Set OPENROUTER_API_KEY environment variable: export OPENROUTER_API_KEY=your-api-key")
            sys.exit(1)

    operation_id = None 
    task = (
        'You are a data scientist, expert in the link prediction algorithms. '
        'Your task is to come up with new innovative link prediction method. '
        'You need to modify this Python code to propose a better link prediction algorithm.'
    )
    prompt0 = task + "\nPropose crazy ideas and only change up to 10 lines of code at the time.\n"
    prompt1 = task + "\nGenerate conventional gradual improvementation for this code here.\n"
    prompt2 = task + "\nPropose a combination of ideas for this problem.\n"
    prompt3 = task + "\nPropose an absolutely crazy idea noone thought of before for this problem.\n"
    prompt4 = task + "\nGenerate a new idea noone can think of and implement it diligently step by step.\n"
    prompt5 = task + "\nGenerate two ideas for this problem and implement a combination of them.\n"
    try:
        # 3. Send Evolution Request
        print("Sending evolution request...")
        
        evolve_response = await client.start_evolution(
            program=program,
            evaluator=evaluator,
            instructions=[prompt0, prompt1, prompt2, prompt3, prompt4, prompt5],
            llm_configs=[
                LLMConfig(
                    model_name=models[0],
                    base_url=base_url,
                    api_key=llm_api_key,
                    llm_client=ClientType.OPENAI,
                    probability=0.9
                ),
                LLMConfig(
                    model_name=models[1],
                    base_url=base_url,
                    api_key=llm_api_key,
                    llm_client=ClientType.OPENAI,
                    probability=0.1
                )
            ],
            max_iterations=max_turns,
            operation_id=operation_id,
            data_path=data_path,
        )
        
        operation_id = evolve_response.operation_id
        print(f"✅ Evolution started. Operation ID: {operation_id}")
        
        # 4. Poll for Status
        print(f"--- Polling status for operation {operation_id} ---")
        for i in range(200): 
            status_data = await client.get_evolution_status(operation_id)
            status = status_data.status
            print(f"   Current status: '{status}'")
            
            if status == 'finished':
                print("✅ Evolution finished!")
                break
            
            if status == 'failed':
                print("❌ Evolution failed.")
                break
            
            await asyncio.sleep(2)
        else:
                print("❌ Experiment timed out waiting for completion.")

        if status_data and getattr(status_data, 'best_program', None):
            print("\n--- Final Program ---")
            print(status_data.best_program)
        else:
            print("❌ Evolution did not complete successfully.")
            print(status_data)
        
    except EvolutionAPIError as e:
        print(f"❌ Evolution API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    


if __name__ == "__main__":
    asyncio.run(main())
