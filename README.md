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

## Training-data preparation

The stable part of Laya retraining is translating and validating the Lean
dataset. Input rows need `text_state` and `tactic`; they may also contain
`row_index`, `theorem`, and a `candidates` list.

```bash
python laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --candidates rw,simp,exact,apply,assumption
```

Each record retains `raw_state`, translated `state`, candidate criteria, the
normalized target tactic, and an audit tag. A server-specific Laya trainer can
be invoked through the existing `module:function` hook:

```bash
python laya_retrain.py \
    data/train.parquet outputs/laya_train.jsonl \
    --trainer my_trainer:train
```

The callable receives `records=...` and `output_path=...`. This boundary is
deliberate because Laya training APIs differ between package/checkpoint
versions; the experiment's data format remains stable.

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
