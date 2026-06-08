#!/usr/bin/env bash
set -euo pipefail

# Find the most recently modified .pkl file in the project root
LATEST_PKL=$(ls -t ./*.pkl 2>/dev/null | head -n 1)

if [[ -z "$LATEST_PKL" ]]; then
    echo "Error: No .pkl files found in the current directory."
    exit 1
fi

echo "Most recent database: $LATEST_PKL"

python tools/extract_best_program.py "$LATEST_PKL"
