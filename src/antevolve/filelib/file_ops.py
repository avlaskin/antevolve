"""Module for file I/O operations."""
import json
import os
from typing import Any


def read_text_file(path: str) -> str:
    """Reads the content of a text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(path: str, content: str, mode: str = 'w'):
    """Writes content to a text file."""
    with open(path, mode, encoding='utf-8') as f:
        f.write(content)


def read_json_file(path: str) -> Any:
    """Reads a JSON file and returns the data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: str, data: Any, indent: int = 2):
    """Writes data to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def append_to_log(path: str, message: str):
    """Appends a message to a log file."""
    with open(path, 'a+', encoding='utf-8') as f:
        f.write(message)
        f.flush()
