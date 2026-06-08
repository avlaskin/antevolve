from typing import List, Any, Optional, Tuple

def running_max(data: List[Optional[float]], program_ids: Optional[List[str]] = None) -> Tuple[List[float], List[str]]:
    """
    Calculates the running maximum for a list of numbers.
    Also returns the program IDs corresponding to the running maximums if provided.
    
    Args:
        data: List of numbers (can contain None).
        program_ids: Optional list of program IDs corresponding to the data.
        
    Returns:
        Tuple of (List of running maximums, List of program_ids of those maximums)
        The lists have length len(data) + 1 (initial 0.0).
    """
    if not data:
        return [], []

    max_values = []
    max_ids = []
    current_max = 0.0
    current_max_id = ""
    # Initial point
    max_values.append(current_max)
    max_ids.append(current_max_id)

    for i in range(len(data)):
        value = data[i]
        if value is None:
            value = 0.0
            
        if value > current_max:
            current_max = value
            if program_ids and i < len(program_ids):
                current_max_id = program_ids[i]
        max_values.append(current_max)
        max_ids.append(current_max_id)

    return max_values, max_ids

def get_running_max_programs(programs: List[Any]) -> List[Any]:
    """
    Extracts programs that set a new running maximum score.
    Assumes programs are already sorted by time/creation if order matters,
    or this function should sort them? 
    Usually caller sorts them. We will assume caller sorts them to be flexible.
    
    Args:
        programs: List of program objects. Must have .scores attribute (dict).
    
    Returns:
        List of programs that improved the score.
    """
    best_programs = []
    current_max = 0.0
    
    for p in programs:
        # Handle various score structures if needed, but standard is p.scores['score']
        score = 0.0
        if hasattr(p, 'scores') and p.scores:
            if 'score' in p.scores:
                score = p.scores['score']
            elif p.scores:
                # Fallback to first value if 'score' key missing but dict not empty
                score = list(p.scores.values())[0]
                
        if score is None:
            score = 0.0
            
        if score > current_max:
            current_max = score
            best_programs.append(p)
            
    return best_programs
