"""Simple evolutionary databse."""

import logging
import hashlib
import pickle
import uuid
import time
import numpy as np

from collections import Counter, defaultdict
from typing import Sequence, Dict, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field

from antevolve.models import Program


def get_latest_file(
        operation_id: str,
        directory: str = ".") -> Optional[Path]:
    """
    Returns the latest modified file in a directory matching a pattern.
    Returns None if no matching files are found.
    """
    dir_path = Path(directory)
    pattern: str = f"db_backup_{operation_id}*"
    
    # 1. Gather all matching files (filtering out directories)
    files = [f for f in dir_path.glob(pattern) if f.is_file()]
    
    # 2. Handle the case where no files match
    if not files:
        return None

    # 3. Find the file with the latest modification time
    #    st_mtime is the standard "Time of Last Modification"
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return latest


def hash_program(p: Program) -> str:
    """Hashes the program."""
    c = p.content['program']
    encoded_string = c.encode('utf-8')
    return hashlib.sha256(encoded_string).hexdigest()

def print_increases(increases: Dict[str, list[float]]):
    logging.info('------ SCORES PER MODEL ---------')
    for k in increases:
        logging.info('Model: %s - Ave. Score delta: %f Scores: %s', k, float(np.average(increases[k])), increases[k])
        print('Model: %s - Ave. Score delta: %f Scores: %s' % (k, float(np.average(increases[k])), increases[k]))
    
        
class EvolutionaryDB:
    """
    An in-memory database for storing and managing evolving programs.
    """

    def __init__(self,
                 num_islands: int | None = 10,
                 crossover_boundary: int = 100):
        """
        Initializes an empty evolutionary database.
        """
        self.programs: Dict[str, Program] = {}
        self.prog_hashes = set()
        self.program_count: int = 0
        self.durations: List[float] = []
        self.backup_prefix: str | None = None
        self.num_islands: int | None = num_islands
        self.crossover_boundary: int | None = crossover_boundary
        self.first_program = None
        self.score_increases: Dict[str, list[float]] = defaultdict(list)
        self.tokens: List[ Dict[str, float | int | bool | str]] = []
        self.evaluator_code: str | None = None

    def track_increase(self, model_name: str, score_delta: float):
        """Tracks score increases."""
        self.score_increases[model_name].append(score_delta)

    def track_tokens(self, scores: Dict[str, str | float | int | bool]):
        """Tracks tokens."""
        self.tokens.append(scores)

    def get_increases(self):
        """Returns score increases object."""
        return self.score_increases
    
    def get_tokens(self):
        """Returns tokens object."""
        return self.tokens

    def get_program_by_id(self, pid: str) -> Optional[Program]:
        """Returns program by id.
        Args:
            pid: program id for program to retrieve.
        
        Returns:
            Program or None if id is not found.
        """
        if pid not in self.programs:
            return None
        return self.programs[pid]

    def add_program(self, program: Program) -> Program:
        """
        Adds a new program to the database.

        If the program does not have a program_id, a new one is generated.

        Args:
            program: The Program object to add.
            island_id: The ID of the island this program belongs to.
            generation: The generation of this program.

        Returns:
            The program with its assigned program_id.
        """
        if program.program_id is None:
            program.program_id = str(uuid.uuid4())

        if program.program_id in self.programs:
            raise ValueError(f"Program with ID {program.program_id} already exists.")
        
        if self.num_islands and program.island_id is None:
            if len(self.programs.values()) < self.num_islands:
                program.island_id =  len(self.programs.values())
            else:
                program.island_id = self.programs[program.parent_ids[-1]].island_id
        
        if not program.metadata:
            program.metadata = {}
        program.metadata['recorded_time'] = time.time()

        phash = hash_program(program)

        if phash in self.prog_hashes:
            print('Program is a duplicate.')
            return program
            
        self.prog_hashes.add(phash)
        self.programs[program.program_id] = program
        self.program_count += 1

        if (self.program_count == 1 or self.program_count % 10 == 0) and self.backup_prefix:
            file_name = self.backup_prefix + str(self.program_count) + '.pkl'
            self.save_to_file(filepath=file_name)
        if not self.first_program:
            self.first_program = program
        return program

    def _combine_scores(self, program: Program) -> float:
        """
        Combines the scores of a program into a single metric.

        This implementation calculates the average of all non-None scores.

        Args:
            program: The program whose scores to combine.

        Returns:
            The combined score. Returns 0.0 if there are no valid scores.
        """
        valid_scores = [s for s in program.scores.values() if s is not None]
        if not valid_scores:
            return 0.0
        if program.metadata and 'feedback' in program.metadata and len(program.metadata['feedback']) > 10:
            valid_scores.append(0.0001 * len(program.metadata['feedback']))  # Feedback gives a slight boost in the
        # TODO: Combine score through multiplication.
        result = sum(valid_scores) / len(valid_scores)
        return result

    def sample_best_programs(self, n: int) -> List[Program]:
        """
        Samples the top n programs based on their combined score.

        Args:
            n: The number of best programs to return.

        Returns:
            A list of the top n Program objects.
        """
        if n <= 0:
            return []

        sorted_programs = sorted(
            self.programs.values(),
            key=self._combine_scores,
            reverse=True
        )
        return sorted_programs[:n]

    def sample_programs(self, n: int, island_id: int = 0) -> List[Program]:
        """
        Samples the top n programs based on their combined score,
        with an island-based strategy.

        Args:
            n: The number of best programs to return.
            island_id: The ID of the current island.
            num_islands: The total number of islands.

        Returns:
            A list of the top n Program objects.
        """
        if n <= 0:
            return []
        
        if not self.num_islands:
            return self.sample_best_programs(n=n)

        if self.reached_crossover():
            # Sample only from the current island
            island_programs = [p for p in self.programs.values() if p.island_id == island_id]
            sorted_programs = sorted(
                island_programs,
                key=self._combine_scores,
                reverse=True
            )
        else:
            # Sample from all islands
            sorted_programs = sorted(
                self.programs.values(),
                key=self._combine_scores,
                reverse=True
            )
        if not sorted_programs:
            new_prog = self.first_program.model_copy()
            new_prog.island_id = island_id
            return [new_prog]

        return sorted_programs[:n]

    def reached_crossover(self) -> bool:
        """Indicates if crossover is necessery."""
        return self.num_islands and len(self.programs) > self.crossover_boundary

    def add_duration(self, duration: float):
        """Stores the duration."""
        self.durations.append(duration)

    def get_all_programs(self) -> List[Program]:
        """
        Returns all programs currently stored in the database.

        Returns:
            A list of all Program objects.
        """
        return list(self.programs.values())
    
    def get_all_durations(self) -> List[float]:
        """
        Returns all durations currently stored in the database.

        Returns:
            A list of all durations.
        """
        return self.durations
    
    
    def show_islands(self):
        """Prints islands for the db."""
        islands = [p.island_id for p in self.programs.values()]
        ci = Counter(islands)
        for i in list(set(islands)):
            print(f' Island {i} has {ci[i]}')
            logging.info(' Island has: %d', ci[i])

    def save_to_file(self, filepath: str):
        """
        Saves the current state of the database to a file.

        Args:
            filepath: The path to the file where the database will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filepath: str) -> 'EvolutionaryDB':
        """
        Loads a database state from a file.

        Args:
            filepath: The path to the file containing the saved database.

        Returns:
            A new EvolutionaryDB instance with the loaded state.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def copy(self) -> 'EvolutionaryDB':
        """
        Creates a deep copy of the database.

        Returns:
            A new, independent copy of the EvolutionaryDB instance.
        """
        new_db = EvolutionaryDB()
        new_db.programs = {pid: p.copy(deep=True) for pid, p in self.programs.items()}
        new_db.program_count = self.program_count
        return new_db

    def __repr__(self) -> str:
        return f"<EvolutionaryDB(program_count={self.program_count})>"

    
def load_per_operation(operation_id: str | None = None) -> EvolutionaryDB | None:
    """Loads the operation if files exist."""
    if not operation_id:
        return EvolutionaryDB()
    filename = get_latest_file(operation_id)
    print('Found database: ', filename)
    if not filename:
        return EvolutionaryDB()
    return EvolutionaryDB.load_from_file(filename)

