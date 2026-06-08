"""Module imports evaluator and checks the score."""

import ast
import os
import datetime
import importlib
import importlib.util
import importlib.metadata

import json
import sys
import time
import subprocess 
import signal
import traceback
import threading
from typing import Tuple, List

from absl import app
from absl import logging
from absl import flags

# TODO: Clen this up, not sure why is this imported locally
from constants import EVAL_FILE, PROG_FILE, SCORE_FILE
from reqinstall import install_for_file, get_installed_packages, filter_std_lib, get_imports_from_file, install_package
from antevolve.filelib import metricslib
from antevolve.filelib import file_ops



FLAGS = flags.FLAGS

flags.DEFINE_string(
    'path', 
    None,
    'Indicates the file and folder where to mutate the program.')

flags.DEFINE_string(
    'model_name', 
    None,
    'Indicates the model to use in the mutation.')

flags.DEFINE_string(
    'base_url', 
    None,
    'Indicates the model base url')

flags.DEFINE_string(
    'api_key', 
    None,
    'Indicates the api key for gemini/model calls.'
)

flags.DEFINE_string(
    'data_path', 
    None,
    'Indicates the data path for the program.'
)

flags.DEFINE_boolean(
    'install_missing',
    False,
    'Whether to install missing packages.'
)

def _save_message(message: str):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    log_message = '\n' + str(current_time) + ' - ' + message
    file_ops.append_to_log('evaluator.txt', log_message)

def load_source(path: str, module_name: str):
    """Imports the module from a file."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None:
            _save_message(f'Error #1 importing {module_name}')
            raise ImportError('Error importing the module.')
        new_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = new_module
        spec.loader.exec_module(new_module)
        print(f'Successfully imported module: {module_name}')
        _save_message(f'Success importing {module_name}')
        return True, new_module

    except ImportError as e:
        print(f"Error importing module : {e}")
        _save_message(f'\nError #2 importing {module_name}\n')
        _save_message(f'\nException importing {e}\n')
        return False, None



# One of key decisions here - how to import and evaluate. 
def evaluate_program(path: str, api_key: str | None = None):
    """Runs program evaluation by loading the program and evaluator."""
    # Dynamically import the module
    prog_file = os.path.join(os.path.dirname(path), PROG_FILE)
    eval_file = os.path.join(os.path.dirname(path), EVAL_FILE)
    #r = install_for_file(prog_file)
    #q = install_for_file(eval_file)
    #if not q or not r:
    #    print('Installing modules failed for ', q, ' or ', r)
    #    return 0.0
    installed = filter_std_lib(get_installed_packages())
    prog_imports = filter_std_lib(get_imports_from_file(prog_file))
    missing = [i for i in prog_imports if i not in installed]
    _save_message('Detecting missing packages...')
    if missing and FLAGS.install_missing:
        _save_message(f'Missing packages: {missing}')
        try:
            for mispackage in missing:
                _save_message(f'Installing ...{mispackage}')
                install_package(mispackage)
                _save_message(f'Installed {mispackage}')
        except Exception as e:
            _save_message(f'Installing failed: {e}')
            _save_message(f'Installing failed: {mispackage}')

    _ = load_source(prog_file, 'program')
    _, evaluator = load_source(eval_file, 'evaluator')
    params = {'data_path': FLAGS.data_path, 'api_key': api_key}
    score = evaluator.evaluate(params) # dict
    print('Score in the sub function: ', score)
    return score


def get_model_config() -> dict[str, str]:
    """Gets model config from the flags."""
    model = FLAGS.model_name
    base_url = FLAGS.base_url
    api_key = FLAGS.api_key
    reasoning = None
    if model and 'gemini' in model:
        reasoning = 'low'
    model_config = {
        'model_name': model,
        'base_url': base_url,
        'api_key': api_key,
        'reasoning': reasoning,
    }
    return model_config


def main(_):
    """Main evaluate logic."""
    _save_message('Evaluator Started.\n')
    path = FLAGS.path
    api_key = FLAGS.api_key
    st = time.monotonic()
    mutation_metrics = metricslib.read_metrics_from_filepath(path)

    _save_message('Checking params...\n')
    if not path:
        logging.error('Path for mutation has to be set.')
        _save_message('No path')
        sys.exit(-1)
    _save_message('\n#2.')
    logging.info('Evaluating a file: %s', path)
    if not os.path.exists(path):
        logging.error('Path passed as flag does not exist.')
        _save_message('Files not found.')
        sys.exit(-2)
    folder = os.path.dirname(path)
    score_file = os.path.join(folder, SCORE_FILE)
    if os.path.exists(score_file):
        os.remove(score_file)
    _save_message(f'\n#3. Checking: {path}')
    model_conf = get_model_config()
    logging.info('Found the model config: %s', model_conf)
    try:
        score = evaluate_program(path, api_key) # IT has to be dict!
        feedback = None
        if score and 'feedback' in score:
            feedback = score['feedback']
            del score['feedback']
        _save_message(f'\n#4. Score: {score}')
    except Exception as e:
        e_as_string = traceback.format_exc()
        _save_message(f'\n#4.5. Exception: {e_as_string}')
        print('Evaluator caught an exception: ', e_as_string)
        score = {'score': 0.00001 }
        feedback = '\nError executing the program: \n' + str(e_as_string)
    elapsed = time.monotonic() - st
    metadata = mutation_metrics
    metadata['evaluation_time'] = elapsed
    if feedback:
        metadata['feedback'] = feedback
    # TODO: Establish a better way to handle errors here.abs
    # Describe a list of errors and their codes.
    if not isinstance(score, dict):
        score = {'score': 0.0}
        metadata['feedback'] = 'Error executing the evaluator, return type is not dict.'
    if 'score' not in score:
        score['score'] = 0.0
        metadata['feedback'] = 'Error executing the evaluator, evaluator does not return score.'
    score['metadata'] = metadata
    _save_message(f'\n Score: {score}')
    with open(score_file, 'w') as fw:
        fw.write(json.dumps(score))

    sys.exit(0)


if __name__ == "__main__":
    app.run(main)
