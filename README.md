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
├── data.py            dataset download, sampling, split carving, audits
├── graph.py           Lean parser, DAG, and PyG conversion
├── gnn.py             model download and public inference API
├── gnn_bundle.py      published GATv2 checkpoint backend
├── laya_interface.py  public Laya adapter exports
├── laya_adapter.py    local Laya loading, translation, and reranking
├── evaluation.py      metrics and JSON reports
├── comparison.py      pool cache, multi-checkpoint comparison, acceptance, run history
├── progress.py        progress bars, heartbeats, and stage banners
├── pipeline.py        download/audit/dedup/pools/split/benchmark CLI
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

The two systems stay separate at inference: the GNN supplies the candidate
set and keeps its own ranking, and Laya ranks that set with its own
probabilities — there is no confidence threshold, fallback, or blending
between them. The benchmark records the selected mode in
`outputs/metrics.json`; `outputs/predictions.json` stores the GNN and Laya
rankings side by side.

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
inference: the row's **own GNN top-10 ranking** (names and scores from
`pipeline.py pools`) as the option list, and a **soft label** that blends the
GNN's score distribution with the gold tactic
(`λ·p_gnn + (1−λ)·onehot(gold)`, `--gnn-lambda`, default 0.3). Rows whose gold
tactic sits outside the GNN top-10 are skipped, never taught a disagreeing
target. This is described in
[How the 10 options are chosen](#how-the-10-options-are-chosen).

The CLI prints stage banners (`pre-flight`, `stage 1/3 translating records`,
`stage 2/3 fine-tuning Laya`, `stage 3/3 comparing checkpoints`) with elapsed
times, so a long run always shows where it is.

### 1. Download the training data

The repository intentionally does not contain the Hugging Face dataset. From
the repository root, download the `train` and `val` splits and combine their
shards in one command each:

```bash
python3 pipeline.py fetch --split train --out data/train.parquet
python3 pipeline.py fetch --split val   --out data/val.parquet
```

`fetch` downloads the parquet shards and concatenates them into the given
file (default `data/<split>.parquet`); add `--no-combine` to keep the raw
shards, and combine them later with `--combine-only`:

```bash
python3 pipeline.py fetch --split train --no-combine
python3 pipeline.py fetch --split train --out data/train.parquet --combine-only
```

The shards live in `data/<owner>--<dataset>/<split>/*.parquet` — the directory
name is the dataset repo id with `/` replaced by `--`, so for this dataset it
is `data/jajostrains--Mathlib-Normalized-Sexpr/train/`. To combine them from
Python instead, use `dataset_shard_paths` so the layout stays in one place:

```bash
python3 - <<'PY'
from pathlib import Path
from data import combine_shards, dataset_shard_paths

paths = dataset_shard_paths(Path("data/jajostrains--Mathlib-Normalized-Sexpr"), "train")
output, rows = combine_shards(paths, Path("data/train.parquet"))
print(f"Wrote {rows} rows to {output}")
PY
```

If Hugging Face reports rate limits, configure a token before downloading:

```bash
export HF_TOKEN="your-hugging-face-token"
```

Do not commit the token. The dataset is ignored by `.gitignore`.

### 2. Audit the splits, then dedup train

Training and evaluation are only meaningful if the native
`train` / `val` / `test` splits share no proof states. The audit streams all
three split files and hashes each row's `text_state` content:

```bash
python3 pipeline.py audit
```

```text
audit: jajostrains/Mathlib-Normalized-Sexpr splits train, val, test
  train  218,727 rows — ... unique states, within-split dupes ...
  ...
  train vs val: DUPLICATES — 28 shared states; 3,985 shared row_index (positional, not duplication)
    state hash examples: ['02af2f53933358a1', ...]
  ...
report: outputs/audit.json
```

Two things are reported per pair, but only one can fail the audit:

- **Shared states** (content hashes) are real contamination — a proof state in
  both train and test makes every held-out metric meaningless. Any overlap
  fails the audit with a non-zero exit and `passed: false` in
  `outputs/audit.json`.
- **Shared `row_index`** is *never* a failure: the index is positional within
  each split (it restarts at 0 in every file), so equal numbers are not equal
  rows. On this dataset the three splits overlap on thousands of indices and
  only a few dozen states — trust the state count, not the index count.

When the audit fails on shared states, write a cleaned train file with
`pipeline.py dedup`. It streams the train split, drops every row whose state
also appears in `val` or `test` (and, by default, repeated states inside
train, keeping the first), writes the result with the original `row_index`
values preserved (so pools still match), and then re-audits the deduped file
— refreshing `outputs/audit.json` so the `laya_retrain.py` gate is satisfied:

```bash
python3 pipeline.py dedup --out data/train.dedup.parquet
```

```text
dedup: train minus states in val, test
  rows 218,727 — kept 217,092, dropped 49 cross-split + 1,586 within-train duplicates (0 blank states kept)
deduped train: data/train.dedup.parquet
  train vs val: 0 shared states
  train vs test: 0 shared states
  val vs test: 0 shared states
audit (deduped): passed
report: outputs/audit.json
audit passed: train.dedup.parquet is disjoint from val, test — use it as the laya_retrain dataset
```

`--keep-within-train-duplicates` restricts the drop to cross-split rows only.
Train on `data/train.dedup.parquet` from here on — the dropped rows are a few
tenths of a percent of train and cost nothing measurable. `laya_retrain.py`
reads `outputs/audit.json` in pre-flight and refuses to run without a passing
report (`--skip-audit` opts out, with a warning that the numbers may be
contaminated). Re-run `audit` (and `dedup`) after any change to the dataset.

### 3. Build the per-row GNN pools

Options and soft labels come from the GNN's own rankings, so score each split
once and store its top-10 ranking and scores per row:

```bash
# resumable; --limit N scores only the first N rows as a smoke test
python3 pipeline.py pools --split train --out data/train-pools.jsonl --device cuda:0
python3 pipeline.py pools --split val   --out data/val-pools.jsonl   --device cuda:0
```

Each JSONL line is
`{"row_index": ..., "target": ..., "ranking": [...], "scores": [...]}` —
the ranking and its score vector are exactly what training will present and
blend (see [How the 10 options are chosen](#how-the-10-options-are-chosen)).

### 4. Create one parquet input file

Step 1 produced `data/train.parquet` (raw) and `data/val.parquet`; step 2
produced the cleaned `data/train.dedup.parquet` — train on that one.
`laya_retrain.py` accepts parquet, JSONL, JSON, or CSV, so point it at
whichever file you prefer. The input must contain at least:

```text
text_state
tactic
```

A `row_index` column is required in practice: it keys the per-row option
shuffle and matches rows to their `--row-pools` entry (canonicalized, so
`123`, `123.0`, and `" 123 "` are the same key). Without it, a stable hash of
the proof state is used instead — but the pools file would then never match,
so supply `row_index`. If the dataset already exists elsewhere, use its
actual path instead of `data/train.dedup.parquet`; the file is not created by
cloning the repo.

### 5. Prepare and preview the Laya records

Run this from the repository root. Preparation does not train the model and
does not require a GPU; it converts each parquet row into one JSONL choice
record whose options are the row's own GNN ranking and whose target is the
soft label:

```bash
python3 laya_retrain.py \
    data/train.dedup.parquet \
    outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl \
    --preview
```

`--row-pools` is **required**: there is no fallback option source. The command
writes `outputs/laya_train.jsonl` and prints the first record. Inspect the
preview before processing the complete dataset. Pre-flight first verifies the
audit, then shows what it will train on:

```text
=== pre-flight checks [+0s] ===
  audit         : outputs/audit.json passed — no duplicate states across train/val/test
  row pools     : 218,727 per-row GNN pools from data/train-pools.jsonl
                 : options and scores come from each row's own ranking; rows without a pool are skipped (pool_missing)
  options/record: up to 10 names from the row's ranking (seed 42, shuffled per row, reshuffled every epoch)
  soft target   : 0.30·GNN + 0.70·gold → gold.tactic.probabilities (gnn-blend-v1)
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
| `pool_missing` | the row has no entry in `--row-pools` (split/key mismatch) |
| `gold_not_in_pool` | the gold tactic is outside the row's GNN top-10 |
| `short_pool` | the row's ranking holds fewer than 2 usable options |
| `bad_target` | the row's GNN scores cannot be blended into a target |

`gold_not_in_pool` is expected for roughly the `1 − GNN recall@10` share of
rows (the benchmark's retrieval ceiling): teaching such a row would disagree
with how the benchmark scores it, so it is skipped and counted. Pass
`--strict` to stop at the first skipped row instead (row numbers are 1-based).
The summary and the sidecar report every count:

```text
outputs/laya_train.stats.json
```

That file records rows read, records written, skipped rows by reason with one
example each, the histogram of options per row (10 unless a GNN ranking is
shorter), `gnn_lambda`, `target_source`, and the mean soft-target gold mass,
and the pools file used. It is read again after training and copied into
`outputs/comparison.json`.

Useful flags for the first run:

```bash
# translate only the first 5000 rows (fast smoke run)
python3 laya_retrain.py data/train.dedup.parquet outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl --limit 5000

# or a reproducible random sample of 5000 rows
python3 laya_retrain.py data/train.dedup.parquet outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl --sample 5000 --seed 42
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
- `questions.tactic.instructions`: the shared prompt (a multiple-choice
  question; answer with a probability per option).
- `questions.tactic.criteria`: a description of each candidate tactic, keyed
  by tactic name.
- `gold.tactic.probabilities`: the **soft training target** —
  `λ·p_gnn + (1−λ)·onehot(gold)` over this row's options.
- `expected.application` / `expected.tactic`: the original tactic application
  (`rw [h]`) and normalized family (`rw`), used for evaluation.
- `metadata`: the shuffled option order, the option count, `gold_in_pool`,
  `target_source`, `distill_lambda`, `target_gold_mass`, the prompt version,
  and whether the row matched its pool.

The translated state preserves Lean identifiers and expressions while expanding
common logical symbols into readable phrases. This gives Laya both exact names
and a clearer description of how the goal relates to the local hypotheses.

### How the 10 options are chosen

Training and inference must present the same *shape* of decision, so a record's
option list **is** the row's GNN ranking — the same top-10 pool the benchmark
reranks at inference:

1. **The pool is the row's own GNN top-k ranking**, stored by
   `pipeline.py pools` (names normalized to canonical tactic families,
   duplicates collapsed, unknown names dropped). Nothing else ever supplies
   options: there is no common-tactic fallback and no distractor top-up.
2. **The gold tactic must already be inside.** If it is not (about
   `1 − GNN recall@10` of rows — the retrieval ceiling), the row is skipped as
   `gold_not_in_pool` and counted, because training it would teach a target
   the benchmark's own scoring contradicts. Gold is never inserted.
3. **A ranking shorter than 10 keeps its natural length.** Short pools are
   recorded (`short_pool` in metadata and the option histogram in the stats
   sidecar), never padded, so training always matches what inference reranks.
4. **Options are shuffled per row** with an RNG keyed on `(seed, row_index)`,
   so position carries no signal, the draws are identical across runs and
   `--limit`/`--sample` subsets, and a second file translated with the same
   pools file reproduces the same choices. Laya additionally reshuffles the
   options every epoch (`shuffle_options`).

#### The soft training label (`--gnn-lambda`)

The gold tactic marks *what is right*; the GNN's scores say *how confident the
retriever was*. The training target keeps both:

```text
gold.tactic.probabilities = λ · p_gnn + (1 − λ) · onehot(gold)
```

- `p_gnn` is the row's GNN score vector normalized over its options (uniform
  when the scores sum to zero; negative or non-finite scores are a
  `bad_target` skip).
- `λ` is `--gnn-lambda` (default **0.3**), so the default target puts 70%
  mass on the gold tactic and 30% on the GNN's distribution — the mean gold
  mass actually written is reported as `mean_gold_mass` in the stats sidecar.
- `λ=0` recovers a hard one-hot label; `λ=1` trains on the pure GNN
  distribution. The blend always sums to 1.

Laya's trainer consumes this directly: with the default `loss="rlcd"` the
soft probabilities act as the cross-entropy target alongside the GRPO term
(`target_from_gold` in `laya.train` renormalizes and validates them), so no
fork of Laya is needed. The prompt matches the framing: Laya answers the
multiple-choice question with a probability for every option that sums to 1 —
"these numbers express how confident you are in the right answer, not how
useful you find an option" — which is exactly what a soft label teaches.

Because the options, the shuffle, and the target all derive from the pools
file, translating a second file (validation) with the **same** pools file and
`--gnn-lambda` presents Laya with an identical distribution.

### 6. Prepare validation data

`--fine-tune --eval-data <path>` fails immediately if that file is missing, and
Laya only reports *overlap* with the training data after it has read and
encoded the whole training set. Prepare the validation JSONL first — it is
cheap to check and expensive to discover late.

Use the dataset's **native `val` split**, never the `test` split, to choose
training settings or a checkpoint (the audit in step 2 already proved the
three splits are disjoint):

```bash
python3 pipeline.py fetch --split val --out data/val.parquet
python3 pipeline.py pools --split val --out data/val-pools.jsonl --device cuda:0
python3 laya_retrain.py \
    data/val.parquet \
    outputs/laya_validation.jsonl \
    --row-pools data/val-pools.jsonl
```

Each side needs its own pools file (`pipeline.py pools --split ...` scores
that split's rows), and both sides must use the same `--gnn-lambda`, so the
held-out metric measures exactly the skill the benchmark measures. The
pre-flight of every fine-tune run intersects `--eval-data` with the training
dataset — matched on `row_index`, falling back to the raw proof state — and
refuses to start when they overlap, because an evaluation over its own
training rows measures memorization. `--allow-eval-overlap` opts out.

`outputs/laya_validation.jsonl` is only the Laya eval file — the benchmark
test split used by the post-training comparison is a different, untouched
split.

The old carve command (`python3 pipeline.py split data/train.parquet`) still
exists for datasets without a native validation split; it writes a disjoint
`data/train_fit.parquet` / `data/validation.parquet` pair from a seeded
assignment. With the native splits in place it is not needed.

### 7. Run fine-tuning on the GPU server

The repository includes a trainer adapter backed by Laya's public
`laya.train.finetune` API. It trains the choice head and encoder with the
`rlcd` loss (GRPO term + soft cross-entropy against `gold.tactic.probabilities`),
fits calibration temperatures, and writes a checkpoint that can be loaded with
`laya.load`. The adapter requires Laya 0.3.28 or newer:

```bash
python3 -m pip install --upgrade --no-deps 'laya>=0.3.28,<0.5'
```

Run this after the training preview and validation preparation succeed. The
first run should be small:

```bash
python3 laya_retrain.py \
    data/train.dedup.parquet \
    outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl \
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
    data/train.dedup.parquet \
    outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl \
    --device cuda:1 \
    --fine-tune \
    --checkpoint-dir outputs/laya-checkpoint \
    --eval-data outputs/laya_validation.jsonl \
    --epochs 4
```

The base checkpoint (`convaiinnovations/laya`), the RLCD loss, the soft-label
blend (`--gnn-lambda 0.3`, the default), the batch shape
(`--micro-batch 8` × `--grad-accum 8`), and the per-epoch option reshuffle are
fixed experiment defaults; the flags to reach for are `--micro-batch` /
`--grad-accum` when the GPU memory budget demands it, `--gnn-lambda` when
sweeping the label blend, and `--model-dir` when fine-tuning from a different
base checkpoint.

Before training starts the command validates the inputs and prints what it is
about to do: the audit result, the pools file and soft-label blend, dataset
row count and expected JSONL size, `--eval-data` existence, schema, and
whether its rows are disjoint from the training dataset, the optimizer budget
(micro-batches per epoch, effective batch, total updates), and whether the
requested `cuda:N` is actually visible. A missing validation file, an
overlapping eval split, a missing/failed audit, or an out-of-range GPU index
fails here, in seconds, instead of after an hour of preprocessing.

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

acceptance (500 rows): FAIL
  FAIL finetuned.recall@5 > gnn.recall@5 (0.410 vs 0.750)
  FAIL finetuned.mrr@10 > gnn.mrr@10 (0.240 vs 0.420)
  ok   finetuned.recall@1 > base.recall@1 (0.130 vs 0.050)
====================================================================
```

and writes:

```text
outputs/comparison.json   full metrics per checkpoint, option counts, acceptance gates, environment
outputs/comparison.png    recall chart plus device and acceptance notes
outputs/history.jsonl     one appended line for this run (including the verdict)
```

The **acceptance verdict** is computed from the same table: the fine-tune is
accepted only when it beats the GNN itself on `recall@5` *and* `mrr@10`, and
beats the base Laya checkpoint on `recall@1`. Each gate records both values,
so a FAIL line is self-explanatory. The intended measurement point is
`--compare-rows 500` (the default `100` is enough for a smoke check but too
small for the verdict); `passed` / `gates` live under `acceptance` in
`outputs/comparison.json` and `outputs/history.jsonl`.

Use `--compare-rows 0` to skip the comparison. It deliberately uses the
benchmark's untouched test split with a fixed seed, not the training or
validation data, so results stay comparable across runs.

The first `benchmark` command also appends to `outputs/history.jsonl` and
refreshes `outputs/environment.json`, so benchmark and fine-tune results are
recorded in the same place.

### Known measurement caveats

- **Pool vs. system recall.** Training only sees rows whose gold tactic is
  *inside* the GNN top-10 (the rest are `gold_not_in_pool` skips), so training
  rows are easier than the full benchmark population. The end-to-end ceiling
  remains `GNN recall@10` (about 0.91 on the test split) times Laya's
  *conditional* rerank accuracy on those 10. Read the conditional columns in
  `outputs/comparison.json` and `history.jsonl`; never read a constructed
  pool's recall@10 as a system metric.
- **Calibration warning.** Loading Laya prints a
  `RuntimeWarning: ... uncalibrated` line about probability adjustment. The
  warnings captured while loading each checkpoint are stored under
  `load_warnings` in `outputs/comparison.json` and `models.<name>.warning` in
  `history.jsonl`. Inference itself never consults a confidence value — Laya's
  probability order is the ranking — so the recall numbers are what to read.
- **Laya on CPU.** If Laya cannot use the requested GPU it falls back to CPU
  silently from its own perspective. The comparison detects this, warns, and
  records `laya_device.requested` / `laya_device.actual` so a slow or
  surprising run is attributable.

### 8. Record and export the run

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
outputs/audit.json                          cross-split duplicate audit (required)
data/train.parquet                          full train split (native split, raw)
data/train.dedup.parquet                    train minus cross-split/within-train state dupes (train on this)
data/val.parquet                            native validation split
data/train-pools.jsonl                      per-row GNN rankings + scores (train)
data/val-pools.jsonl                        per-row GNN rankings + scores (val)
outputs/laya_train.jsonl                    prepared training records
outputs/laya_train.stats.json               rows written, skipped, blend, option counts
outputs/laya_validation.jsonl               held-out records for --eval-data
outputs/laya-checkpoint/                    fine-tuned model
outputs/laya-checkpoint/training_report.json
outputs/laya-checkpoint/run_environment.txt
outputs/comparison.json                     base vs fine-tuned metrics + acceptance
outputs/comparison.png
outputs/history.jsonl                       every benchmark/comparison run
outputs/environment.json                    interpreter, packages, CUDA, git state
```

Copy this frozen checkpoint to the assessment server. Do not retrain or change
the training settings there; use a separate frozen test set for final
evaluation.

### Prompt and candidate descriptions

Inference and retraining use the same prompt and tactic-description catalog
(one source of truth: `DEFAULT_INSTRUCTIONS` in `laya_adapter.py`, imported
by both the trainer and `config.py`). The prompt frames the task as what it
is — **a multiple-choice question over the candidate tactics**: read the goal
and the local hypotheses, eliminate options that cannot apply or make no
progress, keep the ones that advance the goal as written, and answer with a
probability per candidate that sums to 1: "these numbers express how confident
you are in the right answer, not how useful you find an option." Descriptions
explain operations such as rewriting, applying a hypothesis, simplifying,
destructuring, introducing variables, or solving arithmetic. Edit
`DEFAULT_INSTRUCTIONS` to test a different prompt; prepared records stamp it
with `metadata.prompt_version` (`lean-tactic-family-v4-softlabel`) so runs
remain attributable.

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

**1. Audit the splits, then dedup train (seconds–minutes, no GPU)**

```bash
python3 pipeline.py audit
python3 pipeline.py dedup --out data/train.dedup.parquet
python3 pipeline.py fetch --split val --out data/val.parquet
```

`audit` fails on shared *proof states* (the only real contamination); shared
`row_index` values are positional per split and never fail it. `dedup` drops
the offending train rows, writes `data/train.dedup.parquet`, and refreshes
`outputs/audit.json` with a passing report — every later `laya_retrain.py`
run reads that file and trains on the deduped parquet.

**2. Pools (minutes on a GPU, resumable)**

```bash
python3 pipeline.py pools --split train --out data/train-pools.jsonl --limit 2000
```

Confirm the JSONL lines carry `row_index`, `ranking`, and `scores`. Score the
full splits (`--limit` is only the smoke test) before the real preparation.

**3. Preparation smoke run (seconds)**

```bash
python3 laya_retrain.py data/train.dedup.parquet outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl --limit 5000 --preview
```

Check four things in the output: the `audit` and `soft target` pre-flight
lines, the `sample` line (how many of the first 200 rows were usable, and
which reasons were skipped), the skipped-row summary (each reason and count),
and the printed record (`expected.tactic` should be the tactic you expect,
`gold.tactic.probabilities` should sum to 1 with the gold carrying roughly
`0.7 + 0.3·p_gnn`, and `metadata.candidate_order` should list the row's own
GNN ranking options).

**4. Dry run over the whole dataset (minutes, no GPU)**

Remove `--limit`, add `--dry-run`, and let it report the item counts and the
optimizer budget:

```bash
python3 laya_retrain.py data/train.dedup.parquet outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl --dry-run
```

For 218,727 rows the JSONL is about 1 GB (roughly 4.5 KB per record), so the
pre-flight prints the size before writing and warns that fine-tuning reads
that file back into memory. Budget several GB of RAM for the training step
and expect the first progress output only after Laya has read and encoded the
file — that quiet window is what the heartbeat reports on.

**5. Validation split (seconds)**

```bash
python3 pipeline.py pools --split val --out data/val-pools.jsonl
python3 laya_retrain.py data/val.parquet outputs/laya_validation.jsonl \
    --row-pools data/val-pools.jsonl
```

The native `val` split shares no rows with `train` (the audit proved it), and
its own pools file gives it the same option shape as training. Check that the
translated validation file exists and has content — preparing it after
training has started is the classic way to lose an hour.

**6. Fine-tune smoke run (minutes on a GPU)**

```bash
python3 laya_retrain.py data/train.dedup.parquet outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl \
    --device cuda:1 --limit 5000 \
    --fine-tune --checkpoint-dir outputs/laya-checkpoint-smoke \
    --eval-data outputs/laya_validation.jsonl --epochs 1
```

Confirm the pre-flight prints `audit ... passed` and
`eval overlap  : none`, the stage banners appear, the heartbeat or progress
bar starts, a checkpoint directory is written with `training_report.json`,
and the final comparison table (with its acceptance verdict) prints. Delete
the smoke checkpoint afterwards.

**7. Full run**

Repeat step 6 without `--limit` and with `--epochs 4`, then let the automatic
comparison judge the run at the intended measurement size:

```bash
python3 laya_retrain.py data/train.dedup.parquet outputs/laya_train.jsonl \
    --row-pools data/train-pools.jsonl \
    --device cuda:1 \
    --fine-tune --checkpoint-dir outputs/laya-checkpoint \
    --eval-data outputs/laya_validation.jsonl \
    --epochs 4 --compare-rows 500
```

The `acceptance` block in `outputs/comparison.json` is the verdict: all three
gates must pass (beat the GNN on `recall@5` and `mrr@10`, beat base Laya on
`recall@1`). Compare with the base-checkpoint benchmark already stored in
`outputs/history.jsonl`, and re-run the benchmark against the new checkpoint
if you also want the standalone measurement:

```bash
python pipeline.py benchmark --rows 500 --device cuda:1 \
    --laya-model outputs/laya-checkpoint
```

If anything fails, `python pipeline.py validate` plus the pre-flight output
from `laya_retrain.py` localize most problems (columns, tactics, pools keys,
audit, GPU visibility, missing eval file) without touching the GPU.

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
  state), and refuses to start without a passing `outputs/audit.json`
  (`pipeline.py audit`; `--skip-audit` overrides). The native `val` split is
  disjoint by construction, and `--allow-eval-overlap` overrides the
  per-file check when the overlap is intended.
- Every training record is a 10-way decision built from the row's **own GNN
  top-10 ranking** (`pipeline.py pools`, `--row-pools` is required). No
  distractor pool, no gold insertion, no top-up: rows whose gold sits outside
  the ranking are skipped (`gold_not_in_pool`), short rankings keep their
  length, and rows without a pool are skipped (`pool_missing`). The training
  target is the soft blend `λ·p_gnn + (1−λ)·onehot(gold)` written to
  `gold.tactic.probabilities` (`--gnn-lambda`, default 0.3). Options are
  shuffled per row (seeded on `row_index`) and reshuffled every epoch, so
  neither position, ordering, nor option identity leaks the answer, and
  training matches the GNN top-10 reranking shape at inference.
  `outputs/laya_train.stats.json` records the pools file, the per-row option
  counts, the `gold_in_pool` count, `gnn_lambda`, `target_source`, and the
  mean soft-target gold mass. Row pools key on the canonicalized `row_index`
  — `123`, `123.0`, and `" 123 "` are the same key.
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
