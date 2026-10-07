# Math Classifier: GNN and Laya experiment

This repository is the standalone experiment. All source files live directly
in this root directory so it can be cloned and run without an extra project
folder. Run it from the repository root in Google Colab or on a server.

## What it evaluates

```text
Lean text state
    -> text-parser DAG
    -> PyTorch Geometric graph
    -> published GNN ranking
    -> GNN top-10 tactic pool
    -> Laya reranking
    -> recall, MRR, transitions, and saved predictions
```

The comparison is paired: GNN and Laya receive exactly the same top-10 pool.
`recall@10` is the retrieval ceiling; `recall@5` and `recall@1` measure
selection. These are tactic-ranking metrics, not proof-execution success.

## Layout

```text
Math-Classifier/
├── config.py          settings and artifact paths
├── data.py            Hugging Face dataset download and sampling
├── graph.py           Lean parser, DAG, and PyG conversion
├── gnn.py             model download and public inference API
├── gnn_bundle.py      published GATv2 checkpoint backend
├── laya_interface.py  public Laya adapter exports
├── laya_adapter.py    local Laya loading, translation, and reranking
├── evaluation.py      metrics and JSON reports
├── pipeline.py        benchmark and validation CLI
├── laya_retrain.py    translated training-record preparation
└── retraining.py      public retraining exports
```

Generated artifacts belong in `data/`, `weights/`, `cache/`, and `outputs/`.
They are not source files and should not be committed.

## Installation

Create or select a Python environment first. Install a Torch build matching
the server's CUDA setup, then install the experiment:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-laya.txt
```

For a CPU-only machine, use the supplied CPU Torch file instead of choosing a
CUDA wheel:

```bash
python -m pip install -r requirements-torch-cpu.txt
```

Check the installation before downloading the large artifacts:

```bash
python - <<'PY'
import torch
import torch_geometric
import laya
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("pyg: ok")
print("laya: ok")
PY
```

## Google Colab

Run these cells in a fresh Colab runtime:

```python
!git clone <your-math-classifier-repository-url>
%cd <repository>/Math-Classifier
!python -m pip install -e .
!python -m pip install -r requirements.txt
!python -m pip install --no-deps -r requirements-laya.txt
```

Use a GPU runtime for the full benchmark. Start with a small run:

```python
!python pipeline.py validate
!python pipeline.py benchmark --rows 10 --device cuda
```

Only after the smoke run succeeds, increase `--rows` to 500 or more.

The CLI accepts a specific GPU index. List the devices visible to the current
Python environment:

```bash
python3 pipeline.py devices
```

Select GPU 0 or GPU 1 explicitly:

```bash
python3 pipeline.py benchmark --rows 20 --device cuda:0
python3 pipeline.py benchmark --rows 20 --device cuda:1
```

By default, Laya uses the same device as the GNN. To place them separately,
set `--laya-device` explicitly:

```bash
python3 pipeline.py benchmark \
    --rows 500 \
    --device cuda:0 \
    --laya-device cuda:1
```

Laya can use standard scoring or balanced scoring. Balanced mode evaluates
rotated candidate orders and averages the probabilities, which reduces
candidate-position bias but requires more Laya calls:

```bash
python3 pipeline.py benchmark \
    --rows 100 \
    --device cuda:0 \
    --laya-device cuda:0 \
    --laya-mode balanced
```

To keep the GNN ranking when Laya is uncertain, add a confidence threshold.
The threshold is disabled when it is `0`:

```bash
python3 pipeline.py benchmark \
    --rows 500 \
    --device cuda:0 \
    --laya-confidence-threshold 0.60
```

The benchmark records the selected mode, confidence values, and number of
GNN fallbacks in `outputs/metrics.json` and `outputs/predictions.json`.

CPU and automatic selection are also supported:

```bash
python3 pipeline.py benchmark --rows 20 --device cpu
python3 pipeline.py benchmark --rows 20 --device auto
```

Run commands from the `Math-Classifier` repository root. In a Colab notebook,
prefix the same commands with `!`; in a normal terminal, do not use `!`.

## SSH server and Jupyter runtime selection

On an SSH server, `python3`, the Python used by Jupyter, and the Python used
by a shell may be different environments. Install and run the project with the
same interpreter that you intend to use:

```bash
cd /path/to/Math-Classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-laya.txt
```

Check the exact runtime before downloading data or running a benchmark:

```bash
python pipeline.py runtime
python pipeline.py devices
```

`runtime` prints the Python executable, Python version, Torch version,
`CUDA_VISIBLE_DEVICES`, and the GPU indices visible to that interpreter.
Always use the printed interpreter for package installation:

```bash
/path/to/Math-Classifier/.venv/bin/python -m pip install -e .
/path/to/Math-Classifier/.venv/bin/python pipeline.py runtime
```

If you use Jupyter, register that same environment as a kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user \
    --name math-classifier \
    --display-name "Math Classifier (.venv)"
```

Select `Math Classifier (.venv)` as the notebook kernel. In a notebook, verify
that the kernel and shell point to the same Python:

```python
import sys
print(sys.executable)
```

Then run the CLI with that interpreter:

```python
!{sys.executable} pipeline.py runtime
!{sys.executable} pipeline.py devices
!{sys.executable} pipeline.py benchmark --rows 20 --device cuda:0
```

GPU numbering is affected by `CUDA_VISIBLE_DEVICES`. If the server exposes
only physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`, it appears to PyTorch as
`cuda:0`. Use the indices printed by `pipeline.py devices`, not the physical
machine numbering. You can also choose separate devices:

```bash
python pipeline.py benchmark \
    --rows 500 \
    --device cuda:0 \
    --laya-device cuda:1
```

## Benchmark

The default sources are the same links used by the notebook:

```text
model:   jajostrains/Mathlib-Sexpr-GNN
bundle:  pointer-gat-gru
dataset: jajostrains/Mathlib-Normalized-Sexpr
split:   test
Laya:    convaiinnovations/laya
```

Run:

```bash
python pipeline.py benchmark \
    --rows 500 \
    --seed 42 \
    --device cuda
```

The command downloads and verifies the model bundle and dataset shards,
samples rows deterministically, builds text-parser graphs, runs the GNN,
reranks the same top-10 candidates with Laya, and writes:

```text
outputs/metrics.json
outputs/predictions.json
outputs/metrics.png
```

The command also prints a performance summary at the end. Open
`outputs/metrics.png` for a visual comparison of GNN and Laya recall,
conditional recall, and top-1 corrections/regressions. The JSON files remain
available for detailed inspection, but they are not required to understand the
main result.

Use `--work-dir /path/to/run` to keep downloads and results outside the source
tree. The command is intentionally explicit about failures: missing optional
packages, invalid model hashes, malformed states, and invalid Laya responses
stop the run instead of producing incomplete benchmark numbers.

## Laya retraining

There are two separate steps:

```text
Step A: prepare and inspect JSONL records  -> no GPU required
Step B: fine-tune a Laya model            -> GPU/device depends on the trainer
```

`laya_retrain.py` implements Step A and provides the device choice for Step B.
It translates and validates Lean examples in the format consumed by a Laya
trainer. The repository still does not contain a native Laya fine-tuning loop,
but `--device` is accepted by the CLI and passed to the trainer hook.

### 1. Download the training data

The repository intentionally does not contain the Hugging Face dataset. From
the repository root, download the `train` parquet shards:

```bash
python3 - <<'PY'
from pathlib import Path
from data import download_dataset

paths = download_dataset(
    "jajostrains/Mathlib-Normalized-Sexpr",
    "train",
    Path("data/Mathlib-Normalized-Sexpr"),
)

for path in paths:
    print(path)
PY
```

If Hugging Face reports rate limits, configure a token before downloading:

```bash
export HF_TOKEN="your-hugging-face-token"
```

Do not commit the token. The dataset is ignored by `.gitignore`.

### 2. Create one parquet input file

`laya_retrain.py` accepts parquet, JSONL, JSON, or CSV. If the download
produced multiple parquet shards, combine them into the path used below:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

paths = sorted(Path("data/Mathlib-Normalized-Sexpr").rglob("*.parquet"))
if not paths:
    raise SystemExit("No parquet shards found.")

frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
output = Path("data/train.parquet")
output.parent.mkdir(parents=True, exist_ok=True)
frame.to_parquet(output, index=False)

print(f"Wrote {len(frame)} rows to {output}")
print("Columns:", ", ".join(frame.columns))
PY
```

The input must contain at least:

```text
text_state
tactic
```

The optional `row_index`, `theorem`, and `candidates` columns are also
supported. If the dataset already exists elsewhere, use its actual path
instead of `data/train.parquet`; the file is not created by cloning the repo.

### 3. Prepare and preview the Laya records

Run this from `/content/Math-Classifier` in Colab or from the local repository
root:

```bash
python3 laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --candidates rw,simp,simpa,exact,apply,assumption,constructor,intro,cases,rcases,linarith,nlinarith,norm_num,ring,omega,aesop \
    --preview
```

You may also specify the intended trainer device in the same command. The
value is validated immediately and passed to the trainer if `--trainer` is
provided; it does not require Torch when no trainer is provided:

```bash
python3 laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --device cuda:1 \
    --candidates rw,simp,exact,apply,assumption \
    --preview
```

The command writes `outputs/laya_train.jsonl` and prints the first record.
The preview is useful for checking that the dataset columns and state parser
are correct before processing the complete dataset.

This command can run on CPU and does not load Torch, CUDA, or the Laya model:

```bash
python3 laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --candidates rw,simp,simpa,exact,apply,assumption \
    --preview
```

Each record contains:

- `raw_state`: the original Lean proof state.
- `state`: a structured description with `CURRENT GOAL`, `LOCAL HYPOTHESES`,
  and a `TASK` section.
- `questions.tactic.instructions`: the shared proof-state reasoning prompt.
- `questions.tactic.criteria`: descriptions of the candidate tactic actions.
- `expected.application`: the original tactic application, such as `rw [h]`.
- `expected.tactic`: the normalized tactic family, such as `rw`.
- `metadata`: candidate order, candidate count, and prompt version.

The translated state preserves Lean identifiers and expressions while expanding
common logical symbols into readable phrases. This gives Laya both exact names
and a clearer description of how the goal relates to the local hypotheses.

### 4. Run an actual trainer

The repository does not assume a particular Laya fine-tuning API because that
API can differ between Laya versions. To fine-tune, provide a Python module
with a callable named `train`:

```python
# my_trainer.py
def train(records, output_path):
    # Connect this stable record format to the Laya training API
    # for the installed Laya version.
    ...
```

Then run:

```bash
python3 laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --candidates rw,simp,simpa,exact,apply,assumption,constructor,intro,cases,rcases,linarith,nlinarith,norm_num,ring,omega,aesop \
    --device cuda:1 \
    --trainer my_trainer:train
```

The callable receives `records=...`, `output_path=...`, and `device=...`. It is
responsible for loading Laya and performing fine-tuning on that device:

```python
# my_trainer.py
def train(records, output_path, device):
    import laya

    model = laya.load("convaiinnovations/laya", device=device)
    # Call the training API supported by the installed Laya version here.
    print(f"Fine-tuning on {device}")
```

Then run the wrapper with the Python environment that contains Torch, Laya,
and the CUDA-compatible dependencies:

```bash
PYTHON=/home/jovyan/.venvs/notebook-py3.11/bin/python
$PYTHON laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --candidates rw,simp,simpa,exact,apply,assumption \
    --trainer my_trainer:train
```

Supported choices are `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, and other
`cuda:N` values. With `auto`, the trainer receives `cuda` when Torch reports
CUDA availability and `cpu` otherwise. Without `--trainer`, the command only
creates and validates the training JSONL, which is the expected behavior.

### Prompt and candidate descriptions

Inference and retraining use the same prompt and tactic-description catalog.
The prompt asks Laya to analyze the current goal together with local
hypotheses and identify the proof action that best advances that state.
Descriptions explain operations such as rewriting, applying a hypothesis,
simplifying, destructuring, introducing variables, or solving arithmetic.
Use `--instructions "..."` if you need to test a different prompt.

## Important implementation details

- The published vocabulary matches graphs built from `text_state`, not the
  dataset's S-expression columns.
- The model bundle is SHA-256 checked before weights are loaded.
- GNN top-k ties are deterministic because Torch returns a fixed index order
  and Laya ties are sorted by tactic name.
- Laya probabilities must contain exactly the GNN candidate set.
- Ranking evaluation does not execute Lean tactics. Proof execution should be
  added as a separate experiment with timeouts and rollback.

## Development checks

```bash
python pipeline.py validate
python -m compileall -q .
```

The validation command does not require Torch, PyG, Hugging Face access, or
Laya. It checks that the local parser creates a valid proof-state graph.
