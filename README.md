# Text-to-SQL hallucination (KG evaluation)

This repository contains scripts to **prepare prompts**, **run model inference** with [vLLM](https://github.com/vllm-project/vllm), **build a schema knowledge graph** from SQLite databases, and **evaluate** predicted SQL against gold using schema and graph-path metrics.

## Requirements

- **Python 3.10+** (recommended)
- **GPU inference:** `vllm`, `transformers`, CUDA-capable GPUs (see `tensor_parallel_size` below)
- **Data processing:** `tqdm`, `ijson`; optional **Pyserini** if you use `--db_content_index_path` in `process_dataset.py`
- **Evaluation:** `sqlglot` is optional but recommended for `evaluate_kg_path.py` (better SQL parsing; a regex fallback exists if it is not installed)

Install core dependencies (adjust versions for your environment):

```bash
pip install vllm transformers tqdm ijson sqlglot
```

## Repository layout (typical)

- **`process_dataset.py`** — Builds `input_seq` / `output_seq` JSONL-style datasets from raw benchmarks, DB paths, and table metadata.
- **`inference.py`** — Loads a Hugging Face model with vLLM, runs chat prompts, parses ```sql``` blocks, writes JSON with `pred_sqls`.
- **`build_kg.py`** — Scans SQLite files under a root and writes one merged KG JSON (`nodes` / `edges`).
- **`evaluate_kg_path.py`** — Loads a KG JSON and prediction JSON files, reports schema and KG-path / subgraph repair metrics.

Raw data (e.g. `../data/data/dev_sciencebenchmark.json`) and SQLite databases are expected to live **outside** or beside this repo; paths are passed as arguments.

## 1. Prepare prompts (`process_dataset.py`)

Arguments include `--input_data_file`, `--output_data_file`, `--db_path`, `--tables`, `--source`, `--mode`, `--value_limit_num`, and optional `--db_content_index_path`. See the script’s `argparse` block for details and run it from the project root with paths that match your machine.

Example shape (placeholders):

```bash
python process_dataset.py \
  --input_data_file ../data/raw/dev.json \
  --output_data_file ../data/data/dev_sciencebenchmark.json \
  --db_path ../data/databases \
  --tables ../data/tables.json \
  --source <benchmark> \
  --mode <mode> \
  --value_limit_num 3
```

## 2. Inference (`inference.py`)

The entry point is **`inference.py`** (not `infer.py`). It reads a JSON list of examples with an `input_seq` field, generates `n` candidates per example, extracts SQL from fenced ```sql``` blocks, and writes `responses` and `pred_sqls` per row.

Example (Science Benchmark dev, OmniSQL-14B):

```bash
python inference.py \
  --pretrained_model_name_or_path seeklhy/OmniSQL-14B \
  --input_file ../data/data/dev_sciencebenchmark.json \
  --output_file ./results/omnisql14b_sciencebenchmark_dev.json \
  --tensor_parallel_size 4 \
  --n 4 \
  --temperature 1.0
```

- **`--tensor_parallel_size`** — Number of GPUs for tensor parallelism (default in code: `4`).
- **`--n`** — Number of sampled completions per prompt (default: `4`). Evaluation typically uses the first prediction: `pred_sqls[0]`.
- **`--temperature`** — Sampling temperature (default: `1.0`).

Ensure the output directory exists or is creatable; the script creates parent directories for `--output_file` as needed.

## 3. Build a schema KG (`build_kg.py`)

Point `--database-root` at the folder tree that contains your `.sqlite` / `.db` files. Optionally restrict to databases present in a dev split:

```bash
python build_kg.py \
  --database-root ../data/databases \
  --output ./sciencebenchmark_dev_schema_kg.json \
  --dev-json ../data/data/dev_sciencebenchmark.json
```

## 4. Evaluate predictions (`evaluate_kg_path.py`)

Run from the directory where your prediction JSON files live (or adjust paths). Match **`--kg-file`** to the KG built in step 3. **`--input-glob`** selects prediction files.

```bash
cd /path/to/project
python evaluate_kg_path.py \
  --kg-file ./sciencebenchmark_dev_schema_kg.json \
  --input-glob "*sciencebenchmark_dev.json" \
  --output ./results/sciencebenchmark_dev_kg_evaluation_summary.json
```

The summary JSON includes mean metrics over examples (schema, KG path, subgraph repair, etc.) plus parser usage statistics.

## Optional

- **`benchmark_subgraph_repair_report.py`** — Emits a CSV of subgraph repair columns for several benchmark globs; useful for batch comparisons.
- **`evaluate.py`** — Schema-level evaluation without the global KG (separate from `evaluate_kg_path.py`).

## License and data

Benchmark datasets and model weights follow their respective licenses. This README does not redistribute external data; obtain Spider, BIRD, Science Benchmark, or other corpora from their official sources.
