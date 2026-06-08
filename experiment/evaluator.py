"""This is an evaluator qwen v1 here."""
import numpy as np
import os
import random
import json

from typing import Any, Dict, List
import program


def read_json(filename):
    with open(filename, 'r') as fr:
        data = json.loads(fr.read())
    return data

def shuffle_jointly(data, labels):
    """Jointly shuffles train/test data."""
    combined = list(zip(data, labels))
    random.shuffle(combined)
    new_data, new_labels = map(list, zip(*combined))
    return new_data, new_labels

def prep_data(data: dict) -> tuple:
    '''Unwraps the data.'''
    train, tr_labels = shuffle_jointly(data['train_edges'], data['train_labels'])
    test, test_labels = shuffle_jointly(data['test_edges'], data['test_labels'])
    return data['nodes'], train, tr_labels, test, test_labels

_FILESV1 = [
    'faaa304.json',
    'faaa504.json',  
    'faaa704.json',
    'faab304.json',
    'faab504.json',  
    'faab704.json',
    'acrime03.json',
    'acrime05.json',
    'acrime07.json'
] # Max overall expected score: 0.75

_FILESV2 = [
    'faaa304.json',
    'faaa504.json',  
    'faaa704.json',
    'faab304.json',
    'faab504.json',  
    'faab704.json',
    'acrime03.json',
    'acrime05.json',
    'acrime07.json',
    'ajazz_collab03.json',
    'ajazz_collab05.json',
    'ajazz_collab07.json',
] # Max overall expected score: 0.75

_FILESV3 = [
    'faaa304.json',
    'faaa504.json',  
    'faaa704.json', 
    'faab304.json',
    'faab504.json',  
    'faab704.json',
    'acrime03.json',
    'acrime05.json',
    'acrime07.json',
    'ajazz_collab03.json',
    'ajazz_collab05.json',
    'ajazz_collab07.json',
    'abible_nouns03.json',
    'abible_nouns05.json',
    'abible_nouns07.json',
] # Max overall expected score: 0.75

_FILESV4 = [
    'faaa304.json',
    'faaa504.json',  
    'faaa704.json',
    'faab304.json',
    'faab504.json',  
    'faab704.json',
    'acrime03.json',
    'acrime05.json',
    'acrime07.json',
    'ajazz_collab03.json',
    'ajazz_collab05.json',
    'ajazz_collab07.json',
    'abible_nouns03.json',
    'abible_nouns05.json',
    'abible_nouns07.json',
    'alondon_transport03.json',
    'alondon_transport05.json',
    'alondon_transport07.json',  
] # Max overall expected score: 0.75

_FILES = _FILESV4

def evaluate(data: dict) -> dict[str, float | str]:
    """Main evaluation function."""
    mscore = 0.0
    data_path = './data'
    if 'data_path' in data:
        data_path = data['data_path']

    for idx, f in enumerate(_FILES):
        config = program.config_factory()
        model = program.solver_factory()
        data = read_json(os.path.join(data_path, f))
        nodes, train_edge, train_labels, test_edge, test_labels = prep_data(data)
        print('Nodes ', len(nodes), 'Train ', len(train_labels), ' Classes ', sum(train_labels), len(train_labels), ' Test ', sum(test_labels), len(test_labels))
        model.train(config=config, train_edg=train_edge, train_labels=train_labels, real_nodes=nodes)
        predicted = model.predict(test_edge)
        auc = model.compute_auc(predicted=predicted, test_labels=test_labels)
        mscore += abs(float(auc) - 0.5)
        del model
        for i in range(4):
            if idx == i and mscore < (i*0.05):
                print('Early stopping: ')
                break

    return {'score': 2 * mscore / (len(_FILES))  }
