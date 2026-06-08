"""Provides the ability to store additional metrics."""
import os
import json

from antevolve.filelib import file_ops


def get_metrics_path(path: str) -> str:
    """Returns the metrics file path for a given path."""
    file_path = os.path.dirname(path)
    filename = os.path.basename(path)
    if 'eval' in filename:
        filename = 'program.py'
    metrics_name = filename.split('.')[0] + '.json'
    full_metrics_file = os.path.join(file_path, metrics_name)
    return full_metrics_file


def store_metrics_for_filepath(path: str, metrics: dict[str, int | str | float | bool]) -> str:
    """Stores metrics for path."""
    full_metrics_file = get_metrics_path(path)
    file_ops.write_json_file(full_metrics_file, metrics)
    return full_metrics_file


def read_metrics_from_filepath(path: str) -> dict[str, int | str | float | bool]:
    """Reads metrics for path."""
    full_metrics_file = get_metrics_path(path)
    if not os.path.exists(full_metrics_file):
        return {}
    return file_ops.read_json_file(full_metrics_file)
