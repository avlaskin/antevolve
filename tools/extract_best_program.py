
import argparse
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from antevolve.database.database import EvolutionaryDB

def main():
    parser = argparse.ArgumentParser(description="Extract the best program from an EvolutionaryDB pickle file.")
    parser.add_argument("db_file", help="Path to the .pkl database file")
    parser.add_argument("--output", "-o", help="Output filename for the extracted code. Defaults to best_program.py", default="best_program.py")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_file):
        print(f"Error: File '{args.db_file}' not found.")
        sys.exit(1)
        
    print(f"Loading database from {args.db_file}...")
    try:
        # We use the static method to load
        db = EvolutionaryDB.load_from_file(args.db_file)
    except Exception as e:
        print(f"Error loading database: {e}")
        sys.exit(1)
        
    print(f"Database loaded. Contains {db.program_count} programs.")
    
    best_programs = db.sample_best_programs(1)
    
    if not best_programs:
        print("No programs found in the database.")
        sys.exit(0)
        
    best_bp = best_programs[0]
    
    score = 0.0
    if hasattr(best_bp, 'scores') and best_bp.scores:
        # Simple extraction of 'score' key if present, or combine logic
        # Re-using the logic from _combine_scores conceptually or just printing what we have
        score = db._combine_scores(best_bp)
        
    print(f"Best Program ID: {best_bp.program_id}")
    print(f"Score: {score}")
    print(f"Generation: {best_bp.generation}")
    
    content = best_bp.content.get('program', '')
    if not content:
        print("Warning: Program content is empty.")
    
    output_path = args.output
    try:
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Successfully wrote best program to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
