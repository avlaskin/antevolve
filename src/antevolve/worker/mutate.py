"""Key binary for mutating code.

It only works with a single file for now.
"""
import datetime
import os
import random
import time
import signal
import threading
import traceback
import subprocess
import sys


from typing import Tuple, List
from absl import app
from absl import logging
from absl import flags

from antevolve.worker.mutant import mutate_program_on_disk, mutate_program_in_memory
from antevolve.worker.reqinstall import install_for_file
from antevolve.filelib.metricslib import store_metrics_for_filepath
from antevolve.filelib import file_ops


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'path', 
    None,
    'Indicates the file and folder where to mutate the program.')

flags.DEFINE_string(
    'instruction', 
    'You goal is improve existing program.',
    'Indicates the file and folder where to mutate the program.')

flags.DEFINE_string(
    'model_name', 
    'gemini-2.5-flash-lite',
    'Indicates the model to use in the mutation.')

flags.DEFINE_string(
    'base_url', 
    'https://generativelanguage.googleapis.com/v1beta/openai/',
    'Indicates the model base url')

flags.DEFINE_string(
    'api_key', 
    None,
    'Indicates the api key for gemini/model calls.'
)

flags.DEFINE_integer(
    'timeout', 
    600,
    'Sets the max timeout for the operation.'
)

flags.DEFINE_float(
    'temperature', 
    1.0,
    'Sets the temperature for the generation.'
)

flags.DEFINE_integer(
    'max_tokens', 
    64000,
    'Sets the max tokens to reply.'
)

def _save_message(message: str):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    log_message = '\n' + str(current_time) + ' - ' + message
    file_ops.append_to_log('mutator.txt', log_message)

def run_and_await(command: List[str]) -> Tuple[int, str, str]:
    """Runs a command and waits for it to finish."""
    print('Sanity command: ', command)
    _save_message('Started command: %s\n' % command)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err

def terminate_process(time_limit):
    """
    Waits for a specified time and then terminates the current process.
    """
    time.sleep(time_limit)
    print(f"Time limit of {time_limit} seconds reached. Terminating process.")
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        print(f"Error while terminating process: {e}")


def main(_):
    """Core logic."""
    path = FLAGS.path
    api_key = FLAGS.api_key
    timeout = FLAGS.timeout
    st = time.monotonic()
    _save_message('Mutate Started.\n')

    if not path:
        logging.error('Path for mutation has to be set.')
        _save_message('No path')
        sys.exit(-1)

    logging.info('Mutating a file: %s', path)
    if not os.path.exists(path):
        logging.error('Path passed as flag does not exist.')
        _save_message('Files not found.')
        sys.exit(-2)
    _save_message('\n#3.')
    if not api_key:
        logging.warning('API must be passed as flag.')
        _save_message('Api key is missing.')
  
    _save_message('\n#4.')
    instruction = FLAGS.instruction
    model = FLAGS.model_name
    base_url = FLAGS.base_url
    temperature = FLAGS.temperature
    max_tokens = FLAGS.max_tokens
    reasoning = None
    if ('gemini-2.5' in model or 'gemini-3' in model or 'think' in model) and 'lite' not in model:
        reasoning = 'medium'
    if 'sonnet' in model or 'qwen3-coder' in model:
        reasoning = 'high'
        max_tokens = 64_000
    if 'gemini-3-pro' in model or 'gemini-2.5-pro' in model:
        reasoning = 'high'
        max_tokens = 64_000
    model_config = {
        'model_name': model,
        'base_url': base_url,
        'api_key': api_key,
        'reasoning': reasoning,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    _save_message(f'\n#5. Mutating. {path}')
    _save_message(f'\n#6. Model config {model_config}')
    logging.info('Mutation is started.')
    # 1. Mutate a program.
    output_filename = None
    original = ''

    try:
        with open(path, 'r') as fr:
            original = fr.read()
        _save_message(f'\n#7. Mutating  {path}')
        output_filename, ct, pt = mutate_program_on_disk(
            instruction,
            input_file_path=path,
            model_config=model_config,
        )
        _save_message(f'Updated: {output_filename}')
        _save_message(f'Tokens: {ct} Prompt: {pt}')
    except Exception as e:
        _save_message('Error in mutation')
        logging.exception(e)
        print(e)
        e_as_string = traceback.format_exc()
        mutation_time = time.monotonic() - st
        _save_message('Error while mutating: \n %s ' % e_as_string)
        _save_message(str(e))
        sys.exit(-4)

    logging.info('File mutated: %s', output_filename)
    #logging.info('Running imports...')
    #r = install_for_file(output_filename)
    #if not r:
    #    logging.info('Imports for the program are not found.')
    logging.info('#8 Running sanity check...')
    # 2. Check it with pylint
    python_executable_path = sys.executable
    cr, co, ce = run_and_await([python_executable_path, '-m', 'pylint', '--errors-only', output_filename])
    logging.info('Sanity check... gave %d %s %s', cr, co, ce)
    _save_message('\nRunning sanity check.')
    mutation_time = time.monotonic() - st
    if cr != 0:
        logging.info('Sanity check failed. \nReturning previous state and stoping.')
        _save_message('\nSanity check FAILED.')
        store_metrics_for_filepath(
            path, 
            {
                'mutation_time': mutation_time, 
                'passed_sanity_check': False,
                'prompt_tokens': pt,
                'completion_tokens': ct,    
            }
        )
        # 3. Return to previous state if broken. 
        # # We can also next time can try to FIX the issue reported.
        with open(path, 'w') as fw:
            fw.write(original)
            fw.flush()
        sys.exit(-5);
    # We can do more checks here. Like certain functions and so on.
    store_metrics_for_filepath(
        path, 
        {
            'mutation_time': mutation_time, 
            'passed_sanity_check': True,
            'prompt_tokens': pt,
            'completion_tokens': ct,
        }
    )
    _save_message('\nSanity check is a SUCCESS.')
    logging.info('Program is mutated and checked.')
    sys.exit(0);


if __name__ == "__main__":
    app.run(main)
