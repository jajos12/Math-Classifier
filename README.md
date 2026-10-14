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
├── comparison.py      pool cache, multi-checkpoint comparison, run history
├── progress.py        progress bars, heartbeats, and stage banners
├── pipeline.py        dataset split, benchmark and validation CLI
├── laya_retrain.py    translated training-record preparation and CLI
├── laya_trainer.py    built-in laya.train.finetune adapter
├── tactic_choose.py   tactic candidate catalog and name validation
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

The benchmark records the selected mode and the number of GNN fallbacks in
`outputs/metrics.json` and `outputs/predictions.json`.

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
outputs/history.jsonl
outputs/environment.json
```

`metrics.json` and `predictions.json` describe this run. `history.jsonl` gets
one appended line per run (metrics, model, device, package versions) so runs
can be compared later without keeping every `metrics.json` around.
`environment.json` records the interpreter, package versions, CUDA visibility,
and the git commit of the run that produced it.

Use `--laya-model <path>` to rerank the pool with a fine-tuned checkpoint
instead of the published one:

```bash
python pipeline.py benchmark --rows 100 --device cuda:1 \
    --laya-model outputs/laya-checkpoint
```

The command also prints a performance summary at the end. Open
`outputs/metrics.png` for a visual comparison of GNN and Laya recall,
conditional recall, and top-1 corrections/regressions. The JSON files remain
available for detailed inspection, but they are not required to understand the
main result.

The command is intentionally explicit about failures: missing optional
packages, invalid model hashes, malformed states, and invalid Laya responses
stop the run instead of producing incomplete benchmark numbers. Downloads and
results land under the working directory (`weights/`, `data/`, `cache/`,
`outputs/`).

## Laya retraining

There are two separate steps:

```text
Step A: prepare and inspect JSONL records  -> no GPU required
Step B: fine-tune a Laya model            -> GPU/device depends on the trainer
```

`laya_retrain.py` implements both. Step A streams the dataset into
`outputs/laya_train.jsonl` (no GPU, bounded memory) and prints a per-reason
report of every row it could not use. Step B is `--fine-tune`, which calls
Laya's public `laya.train.finetune` through `laya_trainer.py` and then, unless
disabled, reranks a cached GNN pool with both the base and the new checkpoint
so the two can be compared directly.

Every prepared record presents Laya with the same 10-way decision it faces at
inference: the row's gold tactic plus 9 unique distractors drawn at random
from the most common tactic families. This is described in
[How the 10 options are chosen](#how-the-10-options-are-chosen).

The CLI prints stage banners (`pre-flight`, `stage 1/3 translating records`,
`stage 2/3 fine-tuning Laya`, `stage 3/3 comparing checkpoints`) with elapsed
times, so a long run always shows where it is.

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

A `row_index` column is optional but recommended: it keys the per-row
distractor draw (making records reproducible and letting the eval-overlap
guard match rows cheaply). Without it, a stable hash of the proof state is
used instead. If the dataset already exists elsewhere, use its actual path
instead of `data/train.parquet`; the file is not created by cloning the repo.

### 3. Prepare and preview the Laya records

Run this from the repository root. Preparation does not train the model and
does not require a GPU; it converts each parquet row into one JSONL choice
record with a 10-option decision:

```bash
python3 laya_retrain.py \
    data/train.parquet \
    outputs/laya_train.jsonl \
    --preview
```

The command writes `outputs/laya_train.jsonl` and prints the first record.
Inspect the preview before processing the complete dataset. Preparation first
computes the distractor pool and prints it, so you can see the common tactics
it will draw from:

```text
=== pre-flight checks [+0s] ===
  distractor pool: 20 common tactics from data/train.parquet
                 : rw, simp, exact, intro, apply, aesop, ...
  options/record : gold + 9 unique distractors (seed 42, random gold position, reshuffled every epoch)
```

Preparation reads parquet in chunks and writes JSONL row by row, so a
218k-row dataset is never held in memory twice. A `translate` progress bar
shows the row count and rate. Rows that cannot be translated are skipped and
counted instead of aborting the run:

| reason | meaning |
| --- | --- |
| `state_missing` / `state_empty` | `text_state` is null, `NaN`, or blank |
| `state_invalid` | the state could not be parsed into a proof state |
| `tactic_missing` / `tactic_empty` | `tactic` is null or has no usable tactic name |
| `tactic_unknown` | the tactic family is not in `tactic_choose.py` |

Pass `--strict` to stop at the first such row instead (row numbers are
1-based). The summary and the sidecar report every count:

```text
outputs/laya_train.stats.json
```

That file records rows read, records written, skipped rows by reason with one
example each, the histogram of options per row (always `10` unless the pool is
small), how many rows had their gold tactic inside the common pool versus
inserted from outside it, and the distractor pool itself. It is read again
after training and copied into `outputs/comparison.json`.

Useful flags for the first run:

```bash
# translate only the first 5000 rows (fast smoke run)
python3 laya_retrain.py data/train.parquet outputs/laya_train.jsonl --limit 5000

# or a reproducible random sample of 5000 rows
python3 laya_retrain.py data/train.parquet outputs/laya_train.jsonl \
    --sample 5000 --seed 42
```

`--quiet` hides the bar, `--preview` prints the first record after it is
written, and `--dry-run` prepares the records and then asks
`laya.train.dry_run` how many of them become trainable items — without loading
model weights or touching the GPU. `--dry-run` is the cheapest way to find out
that the data is wrong before committing to a training run.

Each record contains:

- `raw_state`: the original Lean proof state.
- `state`: a structured description with `CURRENT GOAL`, `LOCAL HYPOTHESES`,
  and a `TASK` section.
- `questions.tactic.instructions`: the shared proof-state reasoning prompt.
- `questions.tactic.criteria`: a description of each of the 10 candidate
  tactics, keyed by tactic name.
- `expected.application`: the original tactic application, such as `rw [h]`.
- `expected.tactic`: the normalized tactic family, such as `rw`.
- `metadata`: the 10-option order, the option count, whether the gold tactic
  was in the common pool, and the prompt version.

The translated state preserves Lean identifiers and expressions while expanding
common logical symbols into readable phrases. This gives Laya both exact names
and a clearer description of how the goal relates to the local hypotheses.

### How the 10 options are chosen

Training and inference must present the same *shape* of decision, so the
record builder constructs every pool the way the benchmark does at inference
time — a 10-option rerank over plausible tactics:

1. **The pool is the 20 most common tactics.** One projected single-column
   pass over the whole corpus ranks tactic families by frequency
   (normalized to families, ties broken alphabetically). This is the
   `distractor pool` printed in pre-flight and stored in the stats sidecar.
2. **9 distractors are drawn at random, without replacement,** from that pool
   for each row. Uniformly — frequency only selects the pool, it does not
   weight the draw, so a common distractor is not more likely to appear than
   an uncommon one within the pool.
3. **The gold tactic is always included, exactly once.** It is usually one of
   the common tactics already; when a row's gold tactic is *not* in the
   top-20 it is inserted anyway, and the row is counted in
   `gold_inserted` in the stats sidecar.
4. **The gold tactic sits at a uniformly random index.** It is shuffled in
   with the distractors, so it is equally likely to appear in any of the 10
   slots.

Because nothing about the option set or the gold position depends on the row's
*position in the file*, the draws are stable across runs, unaffected by
`--limit`/`--sample`, and reproducible when a second file is translated with
the same pool (`--distractor-pool`, below).

**Nothing else varies.** Every candidate — gold or distractor — gets the same
kind of description and the same state-dependent hints, drawn only from the
proof state. Combined with the random gold position above and the per-epoch
reshuffle inside Laya (the trainer sets `shuffle_options=("choice",)`), this
means the *only* learnable signal is the proof state itself:

- A candidate's **position** carries no information (random per row, reshuffled
  every epoch).
- A candidate's **presence** is drawn from a common-tactic pool, so "the gold is
  always in the list" is not a shortcut — but neither is any distractor more
  positionally or descriptively privileged than the gold.
- The **set of distractors** for a given gold tactic is redrawn per row, so the
  model cannot memorize a fixed "rw competes against {simp, exact, ...}" list.

### 4. Prepare validation data

`--fine-tune --eval-data <path>` fails immediately if that file is missing, and
Laya only reports *overlap* with the training data after it has read and
encoded the whole training set. Prepare the validation JSONL first — it is
cheap to check and expensive to discover late.

Use a separate validation parquet file. Do not use the final assessment/test
file to choose training settings or a checkpoint. If you have no separate
validation split, carve one out of `data/train.parquet` before fine-tuning:

```bash
python3 pipeline.py split data/train.parquet
```

This writes `data/train_fit.parquet` (80% of the rows) and
`data/validation.parquet` (20%) from a seeded assignment: the two files share
no rows, and the split is reproducible (`--validation-fraction` and `--seed`
change it). Fine-tuning then trains on `data/train_fit.parquet`. The
pre-flight of every fine-tune run intersects `--eval-data` with the training
dataset — matched on `row_index`, falling back to the raw proof state — and
refuses to start when they overlap, because an evaluation over its own
training rows measures memorization. `--allow-eval-overlap` opts out.

Translate the validation half, pointing `--distractor-pool` at the *training*
file so both halves present Laya with the identical pool:

```bash
python3 laya_retrain.py \
    data/validation.parquet \
    outputs/laya_validation.jsonl \
    --distractor-pool data/train_fit.parquet
```

`outputs/laya_validation.jsonl` is only the Laya eval file — the benchmark
test split used by the post-training comparison is a different, untouched
split.

### 5. Run fine-tuning on the GPU server

The repository includes a trainer adapter backed by Laya's public
`laya.train.finetune` API. It trains the choice head and encoder, fits
calibration temperatures, and writes a checkpoint that can be loaded with
`laya.load`. The adapter requires Laya 0.3.28 or newer:

```bash
python3 -m pip install --upgrade --no-deps 'laya>=0.3.28,<0.5'
```

Run this after the training preview and validation preparation succeed. The
first run should be small:

```bash
python3 laya_retrain.py \
    data/train_fit.parquet \
    outputs/laya_train.jsonl \
    --device cuda:1 \
    --limit 5000 \
    --fine-tune \
    --checkpoint-dir outputs/laya-checkpoint-smoke \
    --eval-data outputs/laya_validation.jsonl \
    --epochs 1 \
    --dry-run
```

`--dry-run` stops after preparing the records and asking
`laya.train.dry_run` how many become trainable items: rows, valid items,
skipped rows, and the optimizer-step budget, with no model weights loaded.
Drop `--dry-run`, raise `--limit` (or remove it for the full dataset), and set
`--epochs 4` for the real run:

```bash
python3 laya_retrain.py \
    data/train_fit.parquet \
    outputs/laya_train.jsonl \
    --device cuda:1 \
    --fine-tune \
    --checkpoint-dir outputs/laya-checkpoint \
    --eval-data outputs/laya_validation.jsonl \
    --epochs 4
```

The base checkpoint (`convaiinnovations/laya`), the RLCD loss, the batch shape
(`--micro-batch 8` × `--grad-accum 8`), and the per-epoch option reshuffle are
fixed experiment defaults; the flags to reach for are `--micro-batch` /
`--grad-accum` when the GPU memory budget demands it and `--model-dir` when
fine-tuning from a different base checkpoint.

Before training starts the command validates the inputs and prints what it is
about to do: dataset row count and expected JSONL size, `--eval-data`
existence, schema, and whether its rows are disjoint from the training
dataset, the optimizer budget (micro-batches per epoch, effective batch,
total updates), and whether the requested `cuda:N` is actually visible. A
missing validation file, an overlapping eval split, or an out-of-range GPU
index fails here, in seconds, instead of after an hour of preprocessing.

`--fine-tune` prepares `outputs/laya_train.jsonl` and passes that file
directly to `laya.train.finetune` (there is no second copy). During training:

- Laya prints nothing while it reads and encodes the JSONL. A heartbeat line
  appears on stderr if that stage is quiet for 60 seconds, so a slow disk or a
  large file never looks like a hang.
- Once training starts, Laya's `epoch 1/4 step 100 loss 1.234` lines drive a
  progress bar with epoch, step, and loss; `--quiet` shows the raw lines
  instead. Negative RLCD losses (`loss -0.5567`) are normal and update the bar
  like any other value.
- The candidate order inside every record is reshuffled at the start of each
  epoch, so a tactic cannot be identified with a slot; the run prints one line
  confirming this before training starts.
- Budget the wall clock before you start: ~27,300 micro-batches per epoch at
  `--micro-batch 8`, so ~109,000 for 4 epochs over 218k rows. A measured rate
  of ~2.0–2.3 steps/s means roughly **13–15 hours**. The ETA shown during the
  first ~100 steps reflects warmup and overestimates by about 4x — read the
  rate again after step 500.
- Every stage banner carries an elapsed time, and the final report records the
  device actually used and the wall-clock training time in
  `training_report.json`.

The checkpoint directory contains the model files, tokenizer, calibration
configuration, questions, and `training_report.json`. Checkpoint loading can
be verified without running evaluation:

```bash
python3 - <<'PY'
import laya
laya.load("outputs/laya-checkpoint", device="cuda:1")
print("fine-tuned checkpoint loads successfully")
PY
```

Supported devices are `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, and other
`cuda:N` values. The retraining default is `cuda:0`; pass `--device` to select
another GPU or CPU. With `auto`, the trainer selects CUDA when Torch reports
CUDA availability. When a `cuda:N` is requested and `CUDA_VISIBLE_DEVICES` is
not already set, the trainer restricts visibility to that GPU so it becomes
`cuda:0` for Laya; the automatic comparison afterwards re-derives the correct
device automatically. If `CUDA_VISIBLE_DEVICES` is already set, the index is
left alone — it is interpreted relative to that selection, which is what
`pipeline.py devices` prints.

### Automatic comparison after fine-tuning

Unless `--compare-rows 0` is given, the run ends with stage 3/3: the GNN builds
its top-10 pool over `--compare-rows` rows of the test split (cached under
`cache/`, so the GNN runs once and is reused by later runs), and both the base
checkpoint and the new checkpoint rerank that identical pool. The result is a
three-way table — GNN, base Laya, fine-tuned Laya — on the same rows (format
shown, numbers from an example run):

```text
====================================================================
POST-TRAINING COMPARISON (same GNN top-10 pool, same rows)
====================================================================
system           recall@10  recall@5   recall@1    mrr@10
GNN                   0.900     0.750     0.250     0.420
base                  0.350     0.250     0.050     0.140
finetuned             0.520     0.410     0.130     0.240
====================================================================
```

and writes:

```text
outputs/comparison.json   full metrics per checkpoint, option counts, environment
outputs/comparison.png    recall chart plus confidence/fallback notes
outputs/history.jsonl     one appended line for this run
```

Use `--compare-rows 0` to skip the comparison. It deliberately uses the
benchmark's untouched test split with a fixed seed, not the training or
validation data, so results stay comparable across runs.

The first `benchmark` command also appends to `outputs/history.jsonl` and
refreshes `outputs/environment.json`, so benchmark and fine-tune results are
recorded in the same place.

### Known measurement caveats

- **Pool vs. system recall.** The 10-option training pools are *constructed*,
  so pool recall over them is 100% by design — the gold tactic is always one
  of the 10. That number says nothing about the deployed system, which ranks
  the **GNN's** top-10 pool. The end-to-end ceiling is roughly
  `GNN recall@10` (about 0.95 on the test split) times Laya's *conditional*
  rerank accuracy on those 10. Read the conditional columns in
  `outputs/comparison.json` and `history.jsonl`; do not read the constructed
  pool's recall@10 as a system metric.
- **Calibration warning.** Loading Laya prints a
  `RuntimeWarning: ... uncalibrated` line about probability adjustment. It
  means `answer_confidence` is not trustworthy on that checkpoint yet. The
  warnings captured while loading each checkpoint are stored under
  `load_warnings` in `outputs/comparison.json` and `models.<name>.warning` in
  `history.jsonl`. Until a checkpoint is recalibrated, read the recall numbers
  rather than the confidences.
- **Laya on CPU.** If Laya cannot use the requested GPU it falls back to CPU
  silently from its own perspective. The comparison detects this, warns, and
  records `laya_device.requested` / `laya_device.actual` so a slow or
  surprising run is attributable.

### 6. Record and export the run

Keep the checkpoint, prepared data, report, and environment record together
for the separate assessment server:

```bash
{
  echo "date: $(date -Is)"
  echo "git: $(git rev-parse HEAD)"
  echo "python: $(command -v python3)"
  python3 --version
  python3 -m pip freeze
  python3 pipeline.py runtime
  python3 pipeline.py devices
} | tee outputs/laya-checkpoint/run_environment.txt
```

The important files are:

```text
data/train_fit.parquet                 training part of `pipeline.py split`
data/validation.parquet                held-out part of the same split
outputs/laya_train.jsonl              prepared training records
outputs/laya_train.stats.json         rows written, skipped, option counts
outputs/laya_validation.jsonl         held-out records for --eval-data
outputs/laya-checkpoint/              fine-tuned model
outputs/laya-checkpoint/training_report.json
outputs/laya-checkpoint/run_environment.txt
outputs/comparison.json               base vs fine-tuned metrics
outputs/comparison.png
outputs/history.jsonl                 every benchmark/comparison run
outputs/environment.json              interpreter, packages, CUDA, git state
```

Copy this frozen checkpoint to the assessment server. Do not retrain or change
the training settings there; use a separate frozen test set for final
evaluation.

### Prompt and candidate descriptions

Inference and retraining use the same prompt and tactic-description catalog.
The prompt asks Laya to analyze the current goal together with local
hypotheses and identify the proof action that best advances that state.
Descriptions explain operations such as rewriting, applying a hypothesis,
simplifying, destructuring, introducing variables, or solving arithmetic. The
prompt lives in `DEFAULT_INSTRUCTIONS` (`laya_adapter.py`); edit it there to
test a different one.

## Testing a run end-to-end

Work through these in order. Each step is cheap enough to repeat and fails
loudly before the next, more expensive one starts.

**0. Environment (seconds, no downloads)**

```bash
python pipeline.py validate
python -m compileall -q .
python pipeline.py runtime
python pipeline.py devices
```

`validate` proves the parser/graph path works without Torch, PyG, or Laya.

**1. Preparation smoke run (seconds)**

```bash
python3 laya_retrain.py data/train.parquet outputs/laya_train.jsonl \
    --limit 5000 --preview
```

Check four things in the output: the distractor pool pre-flight line, the
`sample` line (how many of the first 200 rows were usable), the skipped-row
summary (each reason and count), and the printed record (`expected.tactic`
should be the tactic you expect, and `metadata.candidate_order` should list
exactly 10 options with the gold tactic among them).

**2. Dry run over the whole dataset (minutes, no GPU)**

Remove `--limit`, add `--dry-run`, and let it report the item counts and the
optimizer budget:

```bash
python3 laya_retrain.py data/train.parquet outputs/laya_train.jsonl --dry-run
```

For 218,727 rows the JSONL is about 1 GB (roughly 4.5 KB per record), so the
pre-flight prints the size before writing and warns that fine-tuning reads
that file back into memory. Budget several GB of RAM for the training step
and expect the first progress output only after Laya has read and encoded the
file — that quiet window is what the heartbeat reports on.

**3. Validation split (seconds)**

```bash
python3 pipeline.py split data/train.parquet
python3 laya_retrain.py data/validation.parquet outputs/laya_validation.jsonl \
    --distractor-pool data/train_fit.parquet
```

Steps 4 and 5 must train on `data/train_fit.parquet`, never on the full
`data/train.parquet`: the pre-flight rejects an `--eval-data` file whose rows
are also in the training dataset, which is exactly the mistake this split
prevents. `--distractor-pool data/train_fit.parquet` makes the validation
records use the training file's common-tactic pool, so training and eval
present Laya with the same option distribution. Check that the translated
validation file exists and has content — preparing it after training has
started is the classic way to lose an hour.

**4. Fine-tune smoke run (minutes on a GPU)**

```bash
python3 laya_retrain.py data/train_fit.parquet outputs/laya_train.jsonl \
    --device cuda:1 --limit 5000 \
    --fine-tune --checkpoint-dir outputs/laya-checkpoint-smoke \
    --eval-data outputs/laya_validation.jsonl --epochs 1
```

Confirm the pre-flight prints `eval overlap  : none`, the stage banners appear,
the heartbeat or progress bar starts, a checkpoint directory is written with
`training_report.json`, and the final comparison table prints. Delete the
smoke checkpoint afterwards.

**5. Full run**

Repeat step 4 without `--limit` and with `--epochs 4`, then re-run the
benchmark against the new checkpoint:

```bash
python pipeline.py benchmark --rows 100 --device cuda:1 \
    --laya-model outputs/laya-checkpoint
```

Compare it with the base-checkpoint benchmark already stored in
`outputs/history.jsonl`. Two runs that differ by more than a few points on
`--rows 100` are usually noise; the automatic comparison in
`outputs/comparison.json` is the paired measurement to trust, because it
holds the pool fixed.

If anything fails, `python pipeline.py validate` plus the pre-flight output
from `laya_retrain.py` localize most problems (columns, tactics, GPU
visibility, missing eval file) without touching the GPU.

## Important implementation details

- The published vocabulary matches graphs built from `text_state`, not the
  dataset's S-expression columns.
- The model bundle is SHA-256 checked before weights are loaded.
- GNN top-k ties are deterministic because Torch returns a fixed index order
  and Laya ties are sorted by tactic name.
- Laya probabilities must contain exactly the GNN candidate set.
- Preparation skips unusable rows and reports them per reason; `--strict`
  restores fail-fast. Both the counts and one example per reason are written
  to `outputs/laya_train.stats.json`.
- Fine-tuning refuses an `--eval-data` file whose rows are also in the
  training dataset (matched on `row_index`, falling back to the raw proof
  state). `pipeline.py split` carves the disjoint pair, and
  `--allow-eval-overlap` overrides the check when the overlap is intended.
- Every training record is a 10-way decision: the gold tactic plus 9 unique
  distractors drawn from the corpus's 20 most common tactics. The gold sits
  at a uniformly random index, distractors are redrawn per row, and the
  option order is reshuffled every epoch — so neither position, ordering, nor
  distractor identity leaks the answer, and training matches the GNN top-10
  reranking shape at inference. `outputs/laya_train.stats.json` records the
  pool, the per-row option counts, and the `gold_in_pool`/`gold_inserted`
  split.
- Every run appends to `outputs/history.jsonl` and writes
  `outputs/environment.json`, so results remain attributable after outputs
  are overwritten.
- Ranking evaluation does not execute Lean tactics. Proof execution should be
  added as a separate experiment with timeouts and rollback.

## Development checks

```bash
python pipeline.py validate
python -m compileall -q .
```

The validation command does not require Torch, PyG, Hugging Face access, or
Laya. It checks that the local parser creates a valid proof-state graph.
`compileall` catches syntax errors across every module, including the ones
that only run on the GPU server.
