"""Mutation service."""

# https://github.com/codelion/openevolve/blob/main/examples/signal_processing/config.yaml

from datetime import datetime
from enum import StrEnum
import logging
import json
import os
import uuid
import random
import shutil
import subprocess
import sys
import argparse
import time
import pydantic
from antevolve.filelib import metricslib, file_ops
import uvicorn


from .constants import *
from fastapi import FastAPI, Depends, Security, HTTPException, status
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader
from fastapi_utils.tasks import repeat_every
from typing import Any, Dict, List, Optional, Tuple


from antevolve.models.llmconfig import LLMConfig
from antevolve.models.mutate import MutateRequest, MutateResponse
from antevolve.models.enums import OperationState
from antevolve.models.mutate import StopServiceRequest, StatusResponse


API_KEY_DEFAULT = 'QWERTY12345'
LLM_API_KEY = 'GOOGLE_API_KEY'
TEMP_FOLDER = 'mut'
DELETE_ON_STOP = True

# Logger config to show the timestamps.
logger = logging.getLogger("uvicorn")
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# --- API Key Security ---

# It's recommended to set the API_KEY in your environment variables.
# For example: export API_KEY="your_super_secret_key"
API_KEY = os.getenv("API_KEY", API_KEY_DEFAULT)
API_KEY_NAME = "X-API-Key"
PERIODIC_TIME = 2
_DEFAULT_MUTATE_TIMEOUT = 300
_DEFAULT_EVALUATE_TIMEOUT = 1200
PYTHON_BINARY = os.getenv("PYTHON_BINARY", "python")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _save_message(message: str):
    """Stores important messages into the file."""
    current_time = datetime.now().strftime('%H:%M:%S')
    with open('worker.txt', 'a+') as fw:
        fw.write('\n' + str(current_time) + ' - ' + message)
        fw.flush()

async def get_api_key(api_header: str = Security(api_key_header)):
    """Dependency to validate the API key from the request header."""
    if api_header == API_KEY:
        return api_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# --- FastAPI Application ---

app = FastAPI(
    title="Mutation Service",
    description="An API to start, stop, and monitor mutation processes.",
    version="1.0.0",
    # Protect all endpoints by default with the API key dependency.
    dependencies=[Depends(get_api_key)]
)
running_processes: Dict[str, Any] = {}

def run_command(command: List[str]) -> Tuple[int, str, str]:
    """Runs and waits for the process."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc

def load_text_file(filename: str) -> Optional[str]:
    """Loads text of the text code file."""
    data = None
    if not os.path.exists(filename):
        return data
    with open(filename, 'r') as fr:
        data = fr.read()
    return data


def run_mutate_background(operation_id: str) -> bool:
    """Runs the process in background."""
    if operation_id not in running_processes:
        logger.error('Attempt to run non existent operation id. %s', operation_id)
        return False
    python_binary = os.environ['PYTHON_BINARY']
    logger.info('Using python binary: %s', python_binary)
    process = running_processes[operation_id]
    instruction = process[REQUEST][INSTRUCTION]
    folder = process[FOLDER]
    filename = os.path.join(folder, PROG_FILE)
    model_name = process[REQUEST][LLM_CONFIG][MODEL_NAME]
    base_url = process[REQUEST][LLM_CONFIG][BASE_URL]
    temperature = process[REQUEST][LLM_CONFIG][TEMPERATURE]
    max_tokens = process[REQUEST][LLM_CONFIG][MAX_TOKENS]
    api_key = process[REQUEST][LLM_CONFIG]['api_key']

    logger.info('Received model config: %s', process[REQUEST][LLM_CONFIG])
    
    # Calculate absolute path to scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mutate_script = os.path.join(current_dir, 'mutate.py')
    logger.info('Mutate script: %s', mutate_script)
    command = [
        python_binary, mutate_script, 
        '--path', filename, 
        '--instruction', instruction,
        '--api_key', api_key,
        '--base_url', base_url,
        '--model_name', model_name,
        '--temperature', str(temperature),
        '--max_tokens', str(max_tokens),
    ]
    logger.info('Starting a mutation job: %s', ' '.join(command))
    proc = run_command(command)
    process[PROCESS_KEY] = proc
    logger.info('Mutation Process is running.')
    return True


# --- Caching ---
DATA_CACHE: Dict[str, str] = {}
DATA_FOLDER = '/app/data'

def prepare_data(remote_path: str) -> str:
    """Downloads and unzips data if needed."""
    if remote_path in DATA_CACHE:
        logger.info('Using cached data for %s -> %s', remote_path, DATA_CACHE[remote_path])
        return DATA_CACHE[remote_path]
    
    # Generate local folder name
    uid = uuid.uuid5(uuid.NAMESPACE_URL, remote_path).hex
    local_folder = os.path.join(DATA_FOLDER, uid)
    
    if os.path.exists(local_folder):
        logger.info('Data folder exists but not in cache: %s', local_folder)
        DATA_CACHE[remote_path] = local_folder
        return local_folder

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        
    logger.info('Preparing data from %s to %s', remote_path, local_folder)
    
    # Download logic (Support local file copy or HTTP/S3)
    zip_path = os.path.join(DATA_FOLDER, f"{uid}.zip")
    
    try:
        if remote_path.startswith('http'):
            import requests # Lazy import
            logger.info('Downloading from URL: %s', remote_path)
            with requests.get(remote_path, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                       f.write(chunk)
        elif remote_path.startswith('s3://'):
            import boto3
            # s3://bucket/key
            parts = remote_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1]
            s3 = boto3.client('s3')
            logger.info('Downloading from S3: %s', remote_path)
            s3.download_file(bucket, key, zip_path)
        else:
             # Assume local path for testing/local mode
             logger.info('Copying from local: %s', remote_path)
             if os.path.exists(remote_path):
                 shutil.copyfile(remote_path, zip_path)
             else:
                 logger.error('Data path not found: %s', remote_path)
                 return ""

        # Unzip
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_folder)
        
        # Cleanup zip
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        DATA_CACHE[remote_path] = local_folder
        return local_folder
        
    except Exception as e:
        logger.error('Failed to prepare data: %s', e)
        return ""

def run_evaluate_background(operation_id: str) -> bool:
    if operation_id not in running_processes:
        logger.error('Attempt to run non existent operation id. %s', operation_id)
        return False
    python_binary = os.environ['PYTHON_BINARY']
    logger.info('Using python binary: %s', python_binary)
    process = running_processes[operation_id]
    # TODO: Think about using this one time.
    
    # Priority: eval_llm_config > llm_config
    if 'eval_llm_config' in process[REQUEST] and process[REQUEST]['eval_llm_config']:
        llm_config = process[REQUEST]['eval_llm_config']
        model_name = llm_config[MODEL_NAME]
        api_key = llm_config['api_key']
        base_url = llm_config[BASE_URL] if BASE_URL in llm_config else None
        logger.info(f'Using specific evaluation LLM config. Model: {model_name}')
    else:
        logger.info('Using NO LLM config for evaluation.')
        # So here we use the very same model that is used for mutation.
        # but technically this strictly is not needed. If it is empty, 
        # the meaning is that evaluator does not use LLM.
        base_url, api_key, model_name = '', '', ''
    
    folder = process[FOLDER]
    eval_file = os.path.join(folder, EVAL_FILE)
    data_path = None
    if 'data_path' in process[REQUEST] and process[REQUEST]['data_path']:
        # Prepare data (download/unzip/cache)
        remote_path = process[REQUEST]['data_path']
        local_data_path = prepare_data(remote_path)
        if local_data_path:
             data_path = local_data_path
        else:
             logger.warning('Data preparation failed for %s', remote_path)

    # Calculate absolute path to scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_script = os.path.join(current_dir, 'evaluate.py')
    logger.info('Evaluate script: %s', evaluate_script)
    command = [
        python_binary, evaluate_script,
        '--path', eval_file, 
    ]
    if base_url:
        command.extend(['--base_url', base_url])
        command.extend(['--model_name', model_name])
        command.extend(['--api_key', api_key])
    if data_path:
        command.extend(['--data_path', data_path])
    logger.info('Starting evaluation job: %s', ' '.join(command))
    proc = run_command(command)
    process[EVAL_PROCESS_KEY] = proc
    logger.info('Evaluation Process is running.')
    return True


@app.on_event("startup")
@repeat_every(seconds=PERIODIC_TIME, logger=logger)  # Runs every 5 seconds
async def periodic_task():
    """Runs periodically."""
    for key, process in running_processes.items():
        if process[STATUS] in {OperationState.FAILED, OperationState.FINISHED}:
            continue
        if PROCESS_KEY not in process:
            continue
        if process[STATUS] == OperationState.STARTED:
            process[STATUS] = OperationState.IN_MUTATION
            proc = None
        elif process[STATUS] == OperationState.IN_MUTATION:
            proc = process[PROCESS_KEY]
            ret_code = proc.poll()
            op_duration = time.time() - process['start_time']
            if ret_code is None:
                if op_duration > _DEFAULT_MUTATE_TIMEOUT:
                    logger.error('Mutation process %s took too long: %f seconds', key, op_duration)
                    process[PROCESS_KEY].kill()
                    process[STATUS] = OperationState.FAILED
                    process[STDERR] = 'Mutation process took too long.'
                    process[STDOUT] = ''
                    process[METADATA] = {MUTATION_TIME: op_duration}
                    process[RETURN_CODE_KEY] = -1
                    process[DURATION] = time.time() - process[START_TIME]
                continue
            if RETURN_CODE_KEY not in process:
                process[RETURN_CODE_KEY] = int(ret_code)
                stdout, stderr = proc.communicate()
                # We opened it in text mode, so stdout is a str
                if ret_code == 0:
                    process[STATUS] = OperationState.IN_EVALUATION
                    del process[RETURN_CODE_KEY]
                    process[EVAL_START_TIME] = time.time()
                    r = run_evaluate_background(key)
                    if not r:
                        process[STATUS] = OperationState.FAILED
                        process[STDERR] = 'Evaluation process failed to start.'
                        process[RETURN_CODE_KEY] = -1
                else:
                    print('Evaluator failed: ', stderr, ' Code: ', ret_code)
                    logger.error('Evaluator failed with code %d. Stderr: %s', ret_code, stderr)
                    process[STATUS] = OperationState.FAILED
                    process[STDERR] = stderr
                    process[STDOUT] = stdout
                folder = process[FOLDER]
                prog_file = os.path.join(folder, PROG_FILE)
                metadata = metricslib.read_metrics_from_filepath(prog_file)
                process[METADATA] = metadata if metadata else {}
                process[METADATA][MUTATION_TIME] = op_duration
                process[DURATION] = time.time() - process[START_TIME]
                proc = None
        elif process[STATUS] == OperationState.IN_EVALUATION:
            proc = process[EVAL_PROCESS_KEY]
            ret_code = proc.poll()
            op_duration = time.time() - process[EVAL_START_TIME]
            if ret_code is None:
                if op_duration > _DEFAULT_EVALUATE_TIMEOUT:
                    logger.error('Evaluation process %s took too long: %f seconds', key, op_duration)
                    process[EVAL_PROCESS_KEY].kill()
                    process[STATUS] = OperationState.FAILED
                    process[STDERR] = 'Evaluation process took too long.'
                    process[STDOUT] = ''
                    folder = process[FOLDER]
                    prog_file = os.path.join(folder, PROG_FILE)
                    metadata = metricslib.read_metrics_from_filepath(prog_file)
                    process[METADATA] = metadata if metadata else {}
                    process[METADATA][EVALUATION_TIME] = op_duration
                    process[RETURN_CODE_KEY] = -1
                    process[DURATION] = time.time() - process[START_TIME]
            else:
                stdout, stderr = proc.communicate()
                folder = process[FOLDER]
                prog_file = os.path.join(folder, PROG_FILE)
                metadata = metricslib.read_metrics_from_filepath(prog_file)
                process[METADATA] = metadata if metadata else {}
                process[METADATA][EVALUATION_TIME] = op_duration
                process[DURATION] = time.time() - process[START_TIME]
                if ret_code == 0:
                    process[STATUS] = OperationState.FINISHED
                    logger.info('Evaluator finished: %s Code: %d', stderr, ret_code)
                    logger.info('Evaluator finished! Stdout: %s', stdout)
                    logger.info('File has length of %d \n', os.path.getsize(os.path.join(process[FOLDER], PROG_FILE)))
                    text = load_text_file(os.path.join(process[FOLDER], PROG_FILE))
                    process[FINAL_PROGRAM] = text
                    try:
                        scores = load_text_file(os.path.join(process[FOLDER], SCORE_FILE))
                        if scores:
                            process[SCORES_KEY] = json.loads(scores)
                            logger.info('Scores as we read them: %s', scores)
                    except Exception as e:
                        logger.error('Error reading scores %s', e)
                        process[SCORES_KEY] = {'score': 0.0000101}

                else:
                    logger.warning('Evaluator failed: %s Code: %d', stderr, ret_code)
                    process[STATUS] = OperationState.FAILED
                    process[STDERR] = stderr
                    process[STDOUT] = stdout

            proc = None




# --- File Operations

def get_time_hh_mm():
    """
    Gets the current time in HH_MM format.

    Returns:
        str: The current time formatted as HH_MM.
    """
    now = datetime.now()
    return now.strftime("_%H_%M")


def create_temp_folder(prefix: str = '/tmp/'):
    """Creates random folder."""
    uid = random.randint(1, 1_000_000_000)
    if not os.path.exists(os.path.join(prefix, TEMP_FOLDER)):
        os.mkdir(os.path.join(prefix, TEMP_FOLDER))
    folder = os.path.join(prefix, TEMP_FOLDER, 'data_' + str(uid) + get_time_hh_mm())
    while os.path.exists(folder):
        uid = random.randint(1, 1_000_000_000)
        folder = os.path.join(prefix, TEMP_FOLDER, 'data_' + str(uid) + get_time_hh_mm())
    logger.info('Creating folder: %s', folder)
    os.mkdir(folder)
    return folder


def save_files_to_folde(
    folder: str,
    program: str,
    evaluator: str) -> bool:
    """Saves programs to folder."""
    file_ops.write_text_file(os.path.join(folder, PROG_FILE), program)
    file_ops.write_text_file(os.path.join(folder, EVAL_FILE), evaluator)
    return True


# --- Endpoints ---
def dict_from_process(process) -> dict[str, str]:
    """Converts operation to a dict for sending over network."""
    scores, final_program, feedback, metadata = None, None, None, None
    duration = None
    if DURATION in process:
        duration = process[DURATION]
    
    if SCORES_KEY in process:
        scores = process[SCORES_KEY]
        final_program = process[FINAL_PROGRAM]
        if METADATA in scores:
            metadata = scores[METADATA]
            del scores[METADATA]
        if FEEDBACK in scores:
            feedback = str(scores[FEEDBACK])
            del scores[FEEDBACK]
    if METADATA in process:
        metadata = process[METADATA]
    return {
        STATUS: process[STATUS],
        START_TIME: process[START_TIME],
        DURATION: duration,
        OPERATION_ID: process[OPERATION_ID],
        SCORES_KEY: scores,
        LLM_CONFIG: process[REQUEST][LLM_CONFIG],
        OLD_SCORES: process[REQUEST][SCORES_KEY],
        FEEDBACK: feedback,
        FINAL_PROGRAM: final_program,
        METADATA: metadata
    }


@app.post("/mutate",
          response_model=MutateResponse,
          summary="Start a new mutation process",
          status_code=status.HTTP_202_ACCEPTED)
async def start_mutation(request: MutateRequest):
    """
    Initiates a new long-running mutation process.
    It returns an `operation_id` to track its status.
    """
    operation_id = uuid.uuid4().hex
    # 1. Create a random folder. Save programs. 
    folder = create_temp_folder()
    r = save_files_to_folde(folder, request.program, request.evaluator)
    if not r:
        return MutateResponse(
            operation_id=operation_id,
            llm_config=request.llm_config,
            status=OperationState.FAILED
        )
    running_processes[operation_id] = {
        STATUS: OperationState.STARTED,
        START_TIME: time.time(),
        REQUEST: request.dict(),
        OPERATION_ID: operation_id,
        FOLDER: folder,
    }
    # 2. Run mutation process first. Store process details.
    if not run_mutate_background(operation_id):
        return MutateResponse(operation_id=operation_id, llm_config=request.llm_config, status=OperationState.FAILED)
    # 3. Run sanity check using pylint.
    # 4. Run evaluation and get the score.
    # 5. Remove random folder.
    # 6. Store mutated program in db.
    return MutateResponse(operation_id=operation_id, llm_config=request.llm_config, status=OperationState.STARTED)

# ---

@app.get("/mutate",
         summary="List all active mutation processes")
async def list_mutations():
    """
    Returns a dictionary of all currently active mutation processes,
    keyed by their `operation_id`.
    """
    if not running_processes:
        return {"message": "No active mutation processes."}
    return [dict_from_process(x) for _ , x in running_processes.items()]

# ---

@app.get("/mutate/{operation_id}",
         summary="Get the status of a specific mutation process")
async def get_mutation_status(operation_id: str):
    """
    Retrieves the current status and details for a specific
    mutation process using its `operation_id`.
    """
    logger.info('Get status %s. Total ops: %d', operation_id, len(running_processes))
    if operation_id not in running_processes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")
    return dict_from_process(running_processes[operation_id])

# ---

def kill_operation(operation_id) -> dict[str, str]:
    """Kills the op and cleans the resources."""
    process = running_processes[operation_id]
    response = dict_from_process(process)
    if process[STATUS] in {OperationState.STARTED, OperationState.IN_MUTATION} and RETURN_CODE_KEY not in process and PROCESS_KEY in process and process[PROCESS_KEY]:
        ## 1. Kill the process first.
        process[PROCESS_KEY].kill()
    if process[STATUS] in {OperationState.IN_EVALUATION} and RETURN_CODE_KEY not in process and EVAL_PROCESS_KEY in process and process[EVAL_PROCESS_KEY]:
        ## 1. Kill the process first.
        process[EVAL_PROCESS_KEY].kill()
    # 2. Delete the folders.
    folder = process[FOLDER]
    if DELETE_ON_STOP:
        shutil.rmtree(folder)
    # 3. Delete operation.
    del running_processes[operation_id]
    return response

@app.delete("/mutate/{operation_id}",
            summary="Stop a specific mutation process")
async def stop_mutation(operation_id: str):
    """
    Stops a single mutation process identified by its `operation_id`.
    Note: The original request defined `StopMutationRequest`, but using a
    path parameter is more conventional for a DELETE operation.
    """
    if operation_id not in running_processes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")
    
    response = kill_operation(operation_id)
    return response

# ---

@app.post("/stop",
          summary="Stop all mutation processes")
async def stop_all_mutations(request: StopServiceRequest):
    """
    Stops all currently running mutation processes if `stop_all` is true.
    """
    if request.stop_all:
        # In a real app, you would iterate and gracefully stop all tasks.
        count = len(running_processes)
        for o in running_processes:
            _ = kill_operation(o)
        return {"message": f"All {count} mutation operations have been stopped."}
    return {"message": "No action taken. Set 'stop_all' to true to stop all operations."}

# ---

@app.get("/status",
         response_model=StatusResponse,
         summary="Get the health status of the service")
async def get_service_status():
    """Health check endpoint to confirm the service is running."""
    return StatusResponse(status=True)

def run():
    """
    Entry point for the worker service.
    Runs the FastAPI app using uvicorn.
    """
    parser = argparse.ArgumentParser(description="Run the worker service.")
    parser.add_argument("--host", default=os.getenv("WORKER_HOST", "0.0.0.0"), help="Host to bind the service to.")
    parser.add_argument("--port", type=int, default=int(os.getenv("WORKER_PORT", "8000")), help="Port to bind the service to.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    parser.add_argument("--python-binary", default=os.getenv("PYTHON_BINARY", "python"), help="Python binary to use for subprocesses.")

    args = parser.parse_args()

    # Allow env var to enable reload if flag is not set
    reload = args.reload or (os.getenv("WORKER_RELOAD", "false").lower() == "true")
    
    # Set the python binary env var so it propagates to the app
    os.environ["PYTHON_BINARY"] = args.python_binary
    
    print(f"Starting worker service on {args.host}:{args.port} with python binary: {args.python_binary}")
    uvicorn.run(
        "antevolve.worker.service:app",
        host=args.host,
        port=args.port,
        reload=reload,
        log_level="info"
    )

# --- Logs ---
@app.get("/logs/err",
         response_class=PlainTextResponse,
         summary="Get worker error log")
async def get_worker_err_log():
    """Returns the content of /var/log/worker.err."""
    log_file = "/var/log/worker.err"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return f.read()
    return "Log file not found."

@app.get("/logs/log",
         response_class=PlainTextResponse,
         summary="Get worker standard log")
async def get_worker_log():
    """Returns the content of /var/log/worker.log."""
    log_file = "/var/log/worker.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return f.read()
    return "Log file not found."


if __name__ == "__main__":
    run()
