"""Module responsible for mutations."""
import datetime
import os
import re
from typing import Optional, List, Tuple

from absl import logging

from antevolve.llmclient.openai_client import OpenAIClient
from antevolve.worker import prompts
from antevolve.filelib import file_ops


def _save_message(message: str, fname: str = 'prompt_trace.txt'):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    log_message = '\n' + str(current_time) + ' - ' + message
    file_ops.append_to_log(fname, log_message)




def parse_code_replacements(text_content: str) -> List[Tuple[str, str]]:
    """
    Parses a string for multiple code replacement blocks and extracts the
    original and replacement code for each block.

    The expected format for each block is:
    <<<<<<< SEARCH
    ... (original code) ...
    ===x===
    ... (replacement code) ...
    >>>>>>> REPLACE

    Args:
        text_content: A string potentially containing multiple formatted blocks.

    Returns:
        A list of tuples, where each tuple contains two strings:
        (original_code, replacement_code). Returns an empty list if no
        blocks are found.
    """
    # Define the regular expression pattern to find all blocks.
    # re.DOTALL allows '.' to match newline characters.
    # The pattern uses non-greedy matching (.*?) to correctly handle multiple
    # blocks.
    think_tag = '</think>'
    if think_tag in text_content:
        # Thinking model, we cut first thinking trace to see the final result:
        position = text_content.index(think_tag)
        text_content = text_content[position:]

    pattern = re.compile(
        r'<<<<<<< SEARCH\s*(.*?)\s*===x===\s*(.*?)\s*>>>>>>> REPLACE',
        re.DOTALL
    )

    # `findall` returns a list of tuples, where each tuple contains the captured groups.
    matches = pattern.findall(text_content)

    if not matches:
        # Fallback to standard pattern for OSS models.
        pattern2 = re.compile(
            r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE',
            re.DOTALL
        )
        matches = pattern2.findall(text_content)

    # Strip leading/trailing whitespace from each captured code block.
    cleaned_matches = [
        (original.strip(), replacement.strip())
        for original, replacement in matches
    ]

    return cleaned_matches


def mutate_program_in_memory(
        task_instruction: str,
        *,
        model_config: dict[str, str],
        input_program: str,
        replacement_instruction: str = prompts.MUTATION_INSTRUCTION) -> str:
    """Mutates a program according to task instructions."""
    query = f"""
# Task
{task_instruction}

# File content
===CONTENT_START===
{input_program}
===CONTENT_END===

{replacement_instruction}
"""
    saved_query = '\n--------------------\n' + query + '\n--------------------\n'
    _save_message(saved_query)  
    saved_query = '\n--- MCONFIG --- n' + str(model_config)  + '\n--------\n'
    _save_message(saved_query)
    r = model_config['reasoning'] if 'reasoning' in model_config else None

    client = OpenAIClient(
        api_key=model_config['api_key'],
        base_url=model_config['base_url'],
    )
    result, ctokens, ptokens = client.generate(
        query,
        reasoning=r,
        model_name=model_config['model_name'],
        temperature=float(model_config['temperature']),
        max_tokens=int(model_config['max_tokens']),
    )
    current_program = input_program
    code_replacements = parse_code_replacements(result)
    for replacement in code_replacements:
        current_program = current_program.replace(
            replacement[0], replacement[1], 1)
    _save_message('\n-----------Generated program-------------\n\n' + current_program)
    _save_message('\n ---- CTOKENS: %d and PTOKENS: %d ----\n' % (ctokens, ptokens))
    return current_program, ctokens, ptokens


def mutate_program_on_disk(
        task_instruction: str,
        *,
        input_file_path: str,
        model_config: dict[str, str],
        output_file_path: Optional[str] = None,
        replacement_instruction: str = prompts.MUTATION_INSTRUCTION_v2):
    """Mutates a program according to task instructions on disk."""
    print('Got a config: ', model_config)
    input_program = file_ops.read_text_file(input_file_path)
    new_program, ctokens, ptokens = mutate_program_in_memory(
        task_instruction,
        model_config=model_config,
        input_program=input_program,
        replacement_instruction=replacement_instruction,
    )
    print('New program: ', new_program)
    final_output_path = input_file_path
    if output_file_path:
        final_output_path = output_file_path
    file_ops.write_text_file(final_output_path, new_program)
    return final_output_path, ctokens, ptokens


def apply_replacement(
        input_file_path: str,
        *,
        original_code: str,
        replacement_code: str,
        output_file_path: str | None = None) -> str | None:
    """
    Applies a code replacement to a new file, keeping the original intact.

    It reads the input file, replaces the first occurrence of the original_code
    with the replacement_code, and writes the content to a new file.

    Args:
        input_file_path: The path to the file to be read.
        original_code: The block of code to be replaced.
        replacement_code: The new block of code.
        output_file_path: Optional. The path for the new file. If not provided,
                          a new filename is generated (e.g., 'file.py' -> 'file.modified.py').

    Returns:
        The path to the newly created file if successful, otherwise None.
    """
    try:
        file_content = file_ops.read_text_file(input_file_path)

        if original_code not in file_content:
            logging.error(
                f"Original code not found in {input_file_path}. No changes made.")
            return None

        # Replace the original code with the new code
        new_content = file_content.replace(original_code, replacement_code, 1)

        # Determine the output file path
        if output_file_path is None:
            base, ext = os.path.splitext(input_file_path)
            final_output_path = f"{base}.modified{ext}"
        else:
            final_output_path = output_file_path

        file_ops.write_text_file(final_output_path, new_content)

        logging.info(
            f"Successfully applied replacement. New file saved to {final_output_path}")
        return final_output_path

    except FileNotFoundError:
        logging.error(f"Error: The file at {input_file_path} was not found.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

