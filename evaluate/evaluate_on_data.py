"""A simple command-line application using absl-py."""
import os

import json
import logging

from absl import app
from absl import flags
from typing import Any

import sys
import pathlib
# Ensure the repo root is on the path so `best` can be imported absolutely
# regardless of how/where this script is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


## Change together start
from best.best_gemini_a import config_factory, solver_factory
PROG = 'geminia'
## Change together end

FLAGS = flags.FLAGS
flags.DEFINE_string('sufix', '', 'Dataset prefix to evaluate. If empty, all prefixes in ./data are used.')
SAMPLED_GRAPHS = 20


def read_from_file(filename):
    """Reads text file."""
    with open(filename, 'r') as fr:
        data = fr.read()
        return data

def load_eval_data(c: str) -> dict[str, Any]:
    """Loads the Eval dataset."""
    files = os.listdir(
        './data/'
    )
    eval_data = {}
    for filen in files:
        filen = './data/' + filen
        if f'{c}' in filen:
            print(' ==> Accepting ', filen)
            fname = filen.split('/')[-1]
            text = read_from_file(filen)
            obj = json.loads(text)
            eval_data[fname] = obj
    print('Keys: ', eval_data.keys())
    return eval_data


def get_unique_prefixes(folder: str = './data') -> list[str]:
    """Returns unique filename prefixes for .jso* files in *folder*.

    A prefix is the part of the filename before the trailing numeric index and
    file extension.  For example ``aeu_airlines00.json`` yields the prefix
    ``aeu_airlines``.

    Args:
        folder: Path to the directory to scan. Defaults to ``'./data'``.

    Returns:
        A sorted list of unique prefix strings found in *folder*.
    """
    import re
    prefixes = set()
    for fname in os.listdir(folder):
        # Match files whose extension starts with .jso (covers .jso, .json, …)
        if re.search(r'\.jso', fname):
            # Strip trailing digits and extension to isolate the prefix
            prefix = re.sub(r'\d+\.jso\w*$', '', fname)
            if prefix:
                prefixes.add(prefix)
    return sorted(prefixes)


def main(argv):
    """The main function for the application.

    Args:
      argv: The command line arguments passed to the binary. 
            The first element is the path to the script, and absl 
            removes all recognized flags before passing the rest.
    """
    folder = './data'
    if FLAGS.sufix:
        prefixes = [FLAGS.sufix]
        print(f'---=== #1 - Using specified prefix: {FLAGS.sufix} ===-')
    else:
        prefixes = get_unique_prefixes(folder)
        print(f'---=== #1 - Found {len(prefixes)} unique prefix(es): {prefixes} ===-')

    print('---=== #2 Calculate ===---')
    for c in prefixes:
        print(f'\n--- Prefix: {c} ---')
        data = load_eval_data(c)
        # Iterate only the files that actually exist for this prefix
        for fname in sorted(data.keys()):
            aucs = []  # Store individual scores.
            print('Processing ', fname)
            config = config_factory()
            model = solver_factory()
            nodes = data[fname]['nodes']
            train_edge = data[fname]['train_edges']
            train_labels = data[fname]['train_labels']
            test_edge = data[fname]['test_edges']
            test_labels = data[fname]['test_labels']
            print('Nodes ', len(nodes), 'Train ', len(train_labels), ' Classes ', sum(train_labels), len(train_labels), ' Test ', sum(test_labels), len(test_labels))
            model.train(config=config, train_edg=train_edge, train_labels=train_labels, real_nodes=nodes)
            predicted = model.predict(test_edge)
            auc = model.compute_auc(predicted=predicted, test_labels=test_labels)
            aucs.append(auc)
            print('Results: ', aucs)
            result_name = f'result_{PROG}_{fname}'
            with open(result_name, 'w') as fw:
                json.dump(aucs, fw)


if __name__ == '__main__':
    app.run(main)
