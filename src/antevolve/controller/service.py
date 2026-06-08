"""Evolution Manager service."""
import datetime
import os
import sys
import uuid
import traceback
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, List
import random
import logging
import httpx
import pickle

import shutil
import argparse
import boto3
from fastapi import FastAPI, Depends, Security, HTTPException, status, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi_utils.tasks import repeat_every

from antevolve.database.database import Program, EvolutionaryDB, load_per_operation
from antevolve.worker.constants import INSTRUCTION, PROGRAM, EVALUATOR
from antevolve.filelib import file_ops
from .worker_client_async import MutationServiceClientAsync
from antevolve.models import EvolveRequest, EvolveIterationsInfo, EvolveResponse, OperationState, LLMConfig
from .models_config import get_models_list


# Defaults suitable for environment variables
_DEFAULT_PORT = 8989
_DEFAULT_LLM_CONFIG = 'llama'

# --- API Key Security ---
# It's recommended to set the API_KEY in your environment variables.
# For example: export API_KEY="your_super_secret_key"
API_KEY_DEFAULT = os.environ.get('CONTROLLER_API_KEY', None)

FILE = './db_backup_'
DB_FOLDER = './data/'
API_KEY = os.getenv("API_KEY", API_KEY_DEFAULT)
API_KEY_NAME = "X-API-Key"
PERIODIC_TIME = 5
ENABLE_FEEDBACK = True
SHOW_PROGS = 3
# Skip evaluation failures
SKIP_LOW = True
# Later in the evo process skip low scores
SKIP_BAD_SCORES = True
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
logger = logging.getLogger(__name__)


def _save_error_message(message: str):
    """Saves the error message to the log file."""
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    file_ops.append_to_log('controller.txt', '\n' + str(current_time) + ' - ' + message)


@dataclass
class ClientCallInfo:
    client: MutationServiceClientAsync
    operation_id: str
    parent_ids: list[str]
    status: str | None = None  # Last client status
    fault_counter: int = 0


def reset_client_info(client_info: ClientCallInfo):
    """Resets the state of the given client_info."""
    client_info.status = None
    client_info.operation_id = ''
    client_info.parent_ids = []
    client_info.fault_counter = 0


@dataclass
class InMemoryInfo:
    database: EvolutionaryDB
    operation_id: str
    root_solution: dict[str, str | list[str]]
    models_list: list[tuple[str, str]] | list['LLMConfig']
    instructions: list[str]
    max_iterations: int = 1
    done_iterations: int = 0
    data_path: str | None = None
    count_positive: bool = True
    clients: list[ClientCallInfo] | None = None
    status: OperationState = OperationState.UNDEFINED
    eval_llm_config: Optional[LLMConfig] = None


async def get_api_key(api_header: str = Security(api_key_header)):
    """Dependency to validate the API key from the request header."""
    if api_header == API_KEY:
        return api_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


# --- FastAPI Application ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup logic goes here."""
    app.active_requests = {}
    logger.info("Application startup: Initializing resources...")
    yield # This separates startup from shutdown logic
    print("Application shutdown: Releasing resources...")
    # TODO: CLEANUP ongoing request if any.


fapp = FastAPI(
    title="Mutation Service",
    description="An API to start, stop, and monitor mutation processes.",
    version="1.0.0",
    #lifespan=lifespan,
    dependencies=[Depends(get_api_key)]
)
fapp.active_requests = {}

def get_response_from_db(app, operation_id: str) -> EvolveResponse:
    """Creates up to date response."""
    info = app.active_requests[operation_id]
    response_info = EvolveIterationsInfo(
        max_iterations=info.max_iterations,
        done_iterations=info.done_iterations,
        total_programs=info.database.program_count,
    )
    best_program = ''
    try:
        best_progs = info.database.sample_best_programs(1)
        if best_progs:
            best_program = best_progs[0].content['program']
    except Exception as e:
        logger.error(f"Error getting best program: {e}")

    return EvolveResponse(
        operation_id=operation_id,
        best_program=best_program,
        iterations_info=response_info,
        status=info.status
    )

def get_workers() -> list[str]:
    """Parses the flag."""
    if 'EVOWORKERS' not in os.environ:
        print('EVOWORKERS must be set to coma separated workers.')
        sys.exit(-1)
    wtext = os.environ['EVOWORKERS']
    workers = [x.strip() for x in wtext.split(',') if len(x) > 2]
    return workers

def crossover_logic(
        main_island_id: int,
        sample: Program,
        db: EvolutionaryDB, 
        num: int) -> tuple[Optional[str], List[str]]:
    """Creates additional instruction with other programs."""
    additional_instruction = None
    extra_parents = []
    if db.reached_crossover():
        # Multi Island sampling
        island_id = 0
        workers_num = len(get_workers())
        if workers_num < 2:

            return None, extra_parents
        while island_id == main_island_id:
            island_id = random.randint(0, workers_num)
        best_progs = db.sample_programs(num, island_id=island_id)
        additional_instruction = ''
        for program in best_progs:
            additional_instruction += (
                '\n# Previously we evaluated this program: '
                '\n# PROGRAM-START'
                f'\n{program.content[PROGRAM]}\n'
                '\n# PROGRAM-END'
                f'\n# This program achieved scores: {program.scores}\n\n'
            )
            extra_parents.append(program.program_id)
    else:
        if not sample.parent_ids:
            return additional_instruction, extra_parents
        l = len(sample.parent_ids)
        if  l > num:
            parents = sample.parent_ids[(l - num):]
        else:
            parents = sample.parent_ids
        # TODO: Rethink this logic here.
        # Now that parent has only one id this code
        # will always add only a sinlge program.
        additional_instruction = ''
        for p in parents:
            program = db.get_program_by_id(p)
            if program:
                additional_instruction += (
                    '\n# Previously we evaluated this program: '
                    '\n# PROGRAM-START'
                    f'\n{program.content[PROGRAM]}\n'
                    '\n# PROGRAM-END'
                    f'\n# This program achieved scores: {program.scores}\n\n'
                )
    return additional_instruction, extra_parents

async def start_single_mutation(info: InMemoryInfo, idx: int):
    """Starts a single mutation."""
    print('Starting new mutation... ', end='')
    r = random.random()
    island_id = 0
    if r > 0.8:
        # 20% probability - pick random island
        island_id = random.randint(4, info.database.num_islands)
        best_progs = info.database.sample_programs(10, island_id=island_id)
    elif r < 0.78:
        # 78% probability - pick idx island
        island_id = idx
        best_progs = info.database.sample_programs(10, island_id=island_id)
    else:
        # 2% probability - start from the begining
        best_progs = [info.database.first_program]
    num_instructions = len(info.instructions)
    sample = random.choice(best_progs)
    client_info = info.clients[idx]
    client = client_info.client
    instruction = info.instructions[idx % num_instructions]
    if ENABLE_FEEDBACK and sample.metadata and 'feedback' in sample.metadata:
        instruction = instruction + (
            f'\n Last program feedback is :\n`{sample.metadata['feedback']}`.\n'
            'Take it into account the feedback for the next code change.\n'
        )
    # Crossover logic  follows here.
    if SHOW_PROGS > 1:
        count = SHOW_PROGS - 1
        addtional_instruction, extra_parents = crossover_logic(island_id, sample=sample, db=info.database, num=count)
        if addtional_instruction is not None:
            instruction += addtional_instruction
    logging.info('\nInstruction: %s \n', instruction)
    current_scores = None
    for k in sample.scores:
        if sample.scores[k]:
            current_scores = sample.scores
    num_progs = len(info.database.get_all_durations())
    # Progressively increase number of tokens
    max_new_tokens = 32_000
    if num_progs > 100:
        max_new_tokens = 40_000
    if num_progs > 500:
        max_new_tokens = 65_536
    if num_progs > 1000:
        max_new_tokens = 84_000
    try:
        client_info.status  = OperationState.IN_PROGRESS
        random_model = random.choice(info.models_list)
        # Using config if available
        temperature = 0.5 + random.random() * 0.5
        start_response = await client.start_mutation(
            program=sample.content[PROGRAM],
            evaluator=sample.content[EVALUATOR],
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            api_key=random_model.api_key,
            program_id=sample.program_id,
            scores=current_scores,
            model_name=random_model.model_name,
            base_url=random_model.base_url,
            temperature=temperature,
            data_path=info.data_path,
            eval_llm_config=info.eval_llm_config # Pass eval config
        )
        client_info.operation_id  = start_response.operation_id
        client_info.status  = start_response.status
        client_info.parent_ids.append(sample.program_id)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        print(f"Network error starting mutation: {e}")
        client_info.fault_counter += 1
        _save_error_message(f"Client {idx} error: {e} Fault counter: {client_info.fault_counter}")
        print('Client ', idx, ' Fault counter: ', client_info.fault_counter)
    except Exception as e:
        print(e)
        str_error = str(e)
        full_traceback_string = traceback.format_exc()
        _save_error_message('\nMutation error: ' + str_error + '\n' + full_traceback_string)


async def initiate_round(
        info: InMemoryInfo,
        workers: list[str]):
    """Creates a round of evolutionary steps."""
    info.clients = []
    pcontent = info.root_solution
    pcontent[INSTRUCTION] = info.instructions[0]
    info.database.add_program(Program(content=pcontent, scores={'score': None}, island_id=0))
    for idx, base_url in enumerate(workers):
        client = MutationServiceClientAsync(base_url=base_url)
        client_info = ClientCallInfo(
            client=client,
            operation_id='',
            parent_ids=[],
            status=OperationState.IN_PROGRESS
        )
        info.clients.append(client_info)
        # 1. START ROUND
        await start_single_mutation(info=info, idx=idx)
    info.status = OperationState.STARTED


async def check_round(info: InMemoryInfo):
    """Checks status of operations."""
    for idx, client_info in enumerate(info.clients):
        if client_info.status:
            operation_id = client_info.operation_id
            if client_info.status == OperationState.IN_PROGRESS:
                # Status state machine:
                # in_progress -> started -> (finished | failed) -> None -> in_progress
                # in progress means we just got the object created
                # and not started the operation yet. So we skip those here.
                continue
            if client_info.status == OperationState.STARTED and client_info.fault_counter > 10:
                try:
                    await client_info.client.stop_mutation(operation_id)
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    print(f"Client {idx} - Timeout/Network error for op {operation_id}: {e}")

                reset_client_info(client_info)
                continue
            try:
                response = await client_info.client.get_mutation_status(operation_id)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                client_info.fault_counter += 1
                print(f"Client {idx} - Timeout/Network error for op {operation_id}: {e}")
                print('Fault counter: ', client_info.fault_counter)
                _save_error_message(f"Client {idx} Fault counter: {client_info.fault_counter} Error: {e}")
                continue
            client_info.status = response.status
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            print(current_time, ' - Operation ', operation_id, ' Status: ', response.status)
            if response.status == 'finished':
                # 2. ROUND FEEDBACK.
                feedback = response.feedback
                duration = response.duration
                info.database.add_duration(duration=duration)
                program = response.final_program
                scores = response.scores
                metadata = response.metadata
                old_scores = response.old_scores
                if not old_scores:
                    old_scores = {'score': 0.0}
                llm_config_used = response.llm_config
                finish_winning = False
                if 'score' in scores and scores['score'] > 0.99:
                    print(f' ======= FOUND SOLUTION -*- {scores} -*- ========== ')
                    print(' ================ Best Program: ================')
                    _progs = info.database.sample_best_programs(10)
                    _isids = [p.island_id for p in _progs]
                    print('====================================================')
                    print(_progs[0].content['program'])
                    print(f' --- SO FAR BEST: {_progs[0].scores} --- IDS: {set(_isids)} ---')
                    finish_winning = True
                print('Found meta ', metadata)
                num_db_progs = len(info.database.get_all_programs())
                if metadata:
                    info.database.track_tokens(metadata)
                if num_db_progs < 3 and info.done_iterations > 100:
                    # Safety switch for the bad evaluators.
                    print('No programs found and done iterations > 100. Stopping.')
                    info.status = OperationState.FINISHED
                    _ = await client_info.client.stop_mutation(operation_id)
                    reset_client_info(client_info)
                    info.done_iterations += 1
                    continue

                if len(program) == len(info.root_solution[PROGRAM]) and program == info.root_solution[PROGRAM]:
                    print('Found same progam')
                    _ = await client_info.client.stop_mutation(operation_id)
                    reset_client_info(client_info)
                    info.done_iterations += 1
                    continue
                # Skip low scores later in the evolutionary process
                # Todo: The number 0.01 here can be automatically adjusted to max / 2  
                if SKIP_BAD_SCORES and num_db_progs > 50 and scores['score'] and scores['score'] < 0.05:
                    print('Found low scoring program. Skipping.')
                    _ = await client_info.client.stop_mutation(operation_id)
                    reset_client_info(client_info)
                    info.done_iterations += 1
                    continue
                # Skip evaluation failures
                if SKIP_LOW and scores['score'] and scores['score'] < 0.000011:
                    print('Found faulty evaluation')
                    _ = await client_info.client.stop_mutation(operation_id)
                    reset_client_info(client_info)
                    info.done_iterations += 1
                    continue                        
                print(f' ======= FOUND SCORE --- {scores} --- ========== ')
                if scores['score']:
                    oscore = 0.0
                    if old_scores and 'score' in old_scores and old_scores['score']:
                        oscore = old_scores['score']
                    print('OLD SCORES FOUND.')
                    info.database.track_increase(llm_config_used.model_name, scores['score'] - oscore)
                else:
                    print('OLD SCORES NOT FOUND.')
                if info.status != OperationState.FINISHED:
                    info.status = OperationState.IN_PROGRESS
                solution = info.root_solution.copy()
                solution[PROGRAM] = program
                model_used = llm_config_used.model_name
                if feedback and isinstance(feedback, str):
                    metadata['feedback'] = feedback
                    metadata['model_name'] = model_used
                    db_program = Program(
                        content=solution,
                        scores=scores,
                        parent_ids=client_info.parent_ids,
                        metadata=metadata,
                    )
                else:
                    metadata['model_name'] = model_used
                    db_program = Program(
                        content=solution,
                        scores=scores,
                        parent_ids=client_info.parent_ids,
                        metadata=metadata,
                    )
                info.database.add_program(db_program)
                info.done_iterations += 1
                reset_client_info(client_info)
                _ = await client_info.client.stop_mutation(operation_id)
                print('DB has ', info.database.program_count, 'Iterations: ', info.done_iterations,' last feedback: ', feedback)
                info.database.show_islands()
                _progs = info.database.sample_best_programs(10)
                _isids = [p.island_id for p in _progs]
                print(f' --- SO FAR BEST: {_progs[0].scores} --- IDS: {set(_isids)} ---')
                if finish_winning:
                    info.status = OperationState.FINISHED
                    info.database.save_to_file(info.database.backup_prefix + 'WIN.pkl')
                continue
            elif response.status == 'failed':
                llm_config_used = response.llm_config
                print('Failed to evaluate the program. ', llm_config_used)
                metadata = response.metadata
                print('Found meta ', metadata)
                if metadata:
                    model_used = llm_config_used.model_name
                    metadata['model_used'] = model_used
                    info.database.track_tokens(metadata)
                _ = await client_info.client.stop_mutation(operation_id)
                reset_client_info(client_info)
                info.done_iterations += 1
                continue
        else:
            if info.status == OperationState.FINISHED:
                continue
            if info.count_positive and len(info.database.get_all_programs()) > info.max_iterations:
                info.status = OperationState.FINISHED
                print('Max Iterations is REACHED.')
                info.database.save_to_file(info.database.backup_prefix + '.pkl')
                print('============= Best Program found: ==================')
                _progs = info.database.sample_best_programs(3)
                print(_progs[0].content['program'])
                print('====================================================')
                continue    
                
            if not info.count_positive and info.done_iterations > info.max_iterations:
                # Starting new calls to evaluate.
                print('Max Iterations is REACHED.')
                info.status = OperationState.FINISHED
                info.database.save_to_file(info.database.backup_prefix + '.pkl')
                print('============= Best Program found: ==================')
                _progs = info.database.sample_best_programs(3)
                print(_progs[0].content['program'])
                print('====================================================')
                continue
            # 3. CONTINUE ROUND.
            await start_single_mutation(info, idx)


@fapp.on_event("startup")
@repeat_every(seconds=PERIODIC_TIME)  # Call this function every 5 seconds
async def periodic_task():
    """Runs periodically."""
    logger.info("Application runs periodic.")
    
    for operation_id, info in fapp.active_requests.items():
        try:
            ws = get_workers()
            if info.status == OperationState.UNDEFINED:
                # Start it first.
                logger.info('Starting the evlution.')
                await initiate_round(fapp.active_requests[operation_id], workers=ws)
            elif info.status == OperationState.STARTED or info.status == OperationState.IN_PROGRESS:
                logger.info('Checking the status.')
                await check_round(info)
            elif info.status == OperationState.FINISHED:
                logger.info('Finishing the evolution.')
                # STOP here all pending OPS!
                for client_info in info.clients:
                   if client_info.operation_id:
                       logger.info('Stopping client operation: %s', client_info.operation_id)
                       try:
                           await client_info.client.stop_mutation(client_info.operation_id)
                       except Exception as e:
                           logger.error('Error stopping mutation: %s', e)
                       reset_client_info(client_info)
                
                # Cleanup from memory to prevent leak, but maybe keep for a bit?
                # For now, let's just mark it cleaned up or similar if we wanted, 
                # but user might want to query result.
                # So we leave it in memory but ensure workers are stopped.
                # To prevent repeated stopping, we can check if we already stopped them.
                # But since we iterate clients, let's clear their operation_id or status?
                # Better: check if client_info.status is not None/Finished.
                
                # Simply clearing clients list might be an option if we don't need them anymore.
                info.clients = []
        except Exception as e:
            print('Internal cycle broken. ', e)
            _save_error_message('PRoblem in inner loop ' + str(e))

@fapp.post("/evolve",
          response_model=EvolveResponse,
          summary="Start a new mutation process",
          status_code=status.HTTP_202_ACCEPTED)
async def create_evolution(request: EvolveRequest):
    """Starts long running operation."""
    print('Request #0')
    if not request.evaluator:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluator code has to be written!"
        )
    print('Request #1.5')
    operation_id = uuid.uuid4().hex
    models_list = get_models_list(request.llm_configs)
    mnames = Counter([x.model_name for x in models_list])
    logger.info(f"Models list: {mnames}")

    database = load_per_operation(request.operation_id)  # Works even when no op_id
    if request.operation_id:
        operation_id = request.operation_id
    database.evaluator_code = request.evaluator
    proc = InMemoryInfo(
        status=OperationState.UNDEFINED,
        database=database,
        root_solution={
            PROGRAM: request.program,
            EVALUATOR: request.evaluator,
        },
        instructions=request.instruction,
        operation_id=operation_id,
        models_list=models_list,
        count_positive=request.count_positive,
        max_iterations=request.max_iterations,
        data_path=request.data_path,
        eval_llm_config=request.eval_llm_config
    )
    print('Request #1 - ', models_list[:2])
    proc.database.backup_prefix = FILE + operation_id + '_'
    fapp.active_requests[operation_id] = proc

    return get_response_from_db(fapp, operation_id=operation_id)

@fapp.get("/evolve/{operation_id}",
          response_model=EvolveResponse,
          summary="Gets the status of the operation")
async def get_evolution(operation_id: str):
    """Starts long running operation."""
    if operation_id not in fapp.active_requests:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")
    return get_response_from_db(fapp, operation_id=operation_id)


@fapp.delete("/evolve/{operation_id}",
            response_model=EvolveResponse,
            summary="Stops the operation")
async def stop_evolution(operation_id: str):
    """Stops the long running operation."""
    if operation_id not in fapp.active_requests:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")
    fapp.active_requests[operation_id].status = OperationState.FINISHED
    return get_response_from_db(fapp, operation_id=operation_id)


@fapp.post("/upload_data",
           summary="Upload a zip file to storage",
           status_code=status.HTTP_201_CREATED)
async def upload_data(file: UploadFile = File(...)):
    """Uploads a .zip file to the configured storage."""
    print(f"Uploading file: {file.filename}")
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed.")
    
    storage_mode = os.getenv("STORAGE_MODE", "aws")
    local_path = os.getenv("LOCAL_STORAGE_PATH", "./data")
    s3_bucket = os.getenv("S3_BUCKET")

    if storage_mode == "local":
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        file_location = os.path.join(local_path, file.filename)
        print(f"Saving to local path: {file_location}")
        try:
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
        except Exception as e:
            print(f"Error saving file locally: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        return {"info": f"file '{file.filename}' saved at '{file_location}'", "data_path": file_location}
    
    elif storage_mode == "aws":
        if not s3_bucket:
             raise HTTPException(status_code=500, detail="S3_BUCKET not configured.")
        print(f"Uploading to S3 bucket: {s3_bucket}")
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_fileobj(file.file, s3_bucket, file.filename)
        except Exception as e:
             print(f"Error uploading to S3: {e}")
             raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")
        return {"info": f"file '{file.filename}' uploaded to s3 bucket '{s3_bucket}'", "data_path": f"s3://{s3_bucket}/{file.filename}"}
    
    else:
        raise HTTPException(status_code=500, detail=f"Unknown storage mode: {storage_mode}")


def run():
    """
    Entry point for the controller service.
    Runs the FastAPI app using uvicorn.
    """
    workers = get_workers()
    print('Found workers.')
    parser = argparse.ArgumentParser(description="Run the controller service.")
    parser.add_argument("--host", default=os.getenv("CONTROLLER_HOST", "0.0.0.0"), help="Host to bind the service to.")
    parser.add_argument("--port", type=int, default=int(os.getenv("CONTROLLER_PORT", str(_DEFAULT_PORT))), help="Port to bind the service to.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    parser.add_argument("--storage-mode", default=os.getenv("STORAGE_MODE", "local"), choices=["local", "aws"],
                        help="Storage mode for uploaded data (local or aws).")
    parser.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", ""), 
                        help="S3 bucket name for aws storage mode.")
    parser.add_argument("--local-storage-path", default=os.getenv("LOCAL_STORAGE_PATH", "./data"), 
                        help="Path for local storage mode.")

    args = parser.parse_args()

    # Allow env var to enable reload if flag is not set
    reload = args.reload or (os.getenv("CONTROLLER_RELOAD", "false").lower() == "true")
    
    
    os.environ["STORAGE_MODE"] = args.storage_mode
    os.environ["S3_BUCKET"] = args.s3_bucket
    os.environ["LOCAL_STORAGE_PATH"] = args.local_storage_path
    
    print(f"Starting controller service on {args.host}:{args.port}")
    print(f"Storage Mode: {args.storage_mode}")
    import uvicorn
    uvicorn.run(
        "antevolve.controller.service:fapp",
        host=args.host,
        port=args.port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run()
