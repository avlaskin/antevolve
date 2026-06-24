# Evaluate on Data

Runs a link-prediction evaluation over **every dataset** found in the `./data`
folder and writes per-file AUC results as JSON files in the current directory.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python evaluate_on_data.py
```

The script automatically:

1. Calls `get_unique_prefixes('./data')` to discover all dataset prefixes
   (e.g. `aeu_airlines`, `aeuroroad`).
2. Loads all matching files for each prefix via `load_eval_data()`.
3. Trains and evaluates the model on every file, writing results to
   `result_<PROG>_<filename>.json` in the working directory.

No flags are required. The legacy `--sufix` / `--gindex` flags are no longer
used by `evaluate_on_data.py`.

### Example — run on a single prefix (using `main.py`)

```bash
python main.py --sufix='aeu_airlines'
```

## Data folder layout

Place dataset files in `./data/` using the naming convention:

```
./data/<prefix><NN>.json
```

**Example:**

```
./data/aeu_airlines00.json
./data/aeu_airlines01.json
./data/aeuroroad00.json
./data/aeuroroad01.json
```

Each JSON file must contain the keys:
`nodes`, `train_edges`, `train_labels`, `test_edges`, `test_labels`.

## Choosing an algorithm

The algorithm is imported from the `../best/` folder. All modules there expose
the same interface (`config_factory`, `solver_factory`), so switching is a
one-line change at the top of the script:

```python
# evaluate_on_data.py  —  line 17
from best.best_gemini_a import config_factory, solver_factory  # default
```

| Module | Description |
|---|---|````

| `best_gemini_a` | Gemini-based solver, variant A (default) |
| `best_gemini_b` | Gemini-based solver, variant B |
| `best_gemini_2k` | Gemini-based solver, 2000 iterations. |
| `best_qwen_a`   | Qwen-based solver, variant A |
| `best_qwen_b`   | Qwen-based solver, variant B |

Replace the module name in the import and set `PROG` accordingly so result
filenames reflect the algorithm used:

```python
from best.best_qwen_a import config_factory, solver_factory

PROG = 'qwena'
```

## Output

For each processed file a result is written to the working directory:

```
result_<PROG>_<source_filename>
```

**Example:** evaluating `aeu_airlines03.json` with the default `geminia` solver
produces `result_geminia_aeu_airlines03.json` containing a list with one AUC
score.