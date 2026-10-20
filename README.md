# Russian Cross-Encoder Reranker

End-to-end training pipeline and trained-checkpoint recipe for a
Russian-language cross-encoder reranker. The repository contains the
full data-processing, training, and evaluation code that produced the
results in the accompanying thesis (`thesis.pdf`).

## What's in the repository

| Folder         | What it contains                                                       |
|----------------|------------------------------------------------------------------------|
| `data_prep/`   | Off-pipeline helpers for normalising raw sources to the input schema   |
| `pipeline/`    | The six-stage training pipeline (splitting → training)                 |
| `evaluation/`  | Two-stage BM25 + reranker evaluation on Russian IR benchmarks          |
| `configs/`     | Per-stage TOML configurations; `configs/thesis/` reproduces the paper  |
| `docs/`        | `PIPELINE.md`, `REPRODUCING_THESIS.md`, `REPRODUCING_ABLATIONS.md`     |
| `examples/`    | Self-contained miniature run to verify the install                     |
| `utilities/`   | Off-pipeline utilities                                                 |

For a narrative walkthrough of the pipeline, read `docs/PIPELINE.md`.

## Requirements

- Python 3.13+
- CUDA 12+ (for FAISS-GPU and distributed training)
- An OpenAI-compatible embedding endpoint
- A single GPU is enough to run `examples/`; the thesis training
  configuration uses 8 GPUs. Detailed hardware notes are in
  `docs/REPRODUCING_THESIS.md`.

The full dependency list is in `pyproject.toml`.

## Install

    uv sync
    # or: pip install -e .

## Configure

    cp .env.example .env
    # Edit .env to point at your data directories, embedding API,
    # and base model

Required environment variables are documented in `.env.example`.
Hyperparameters used for the thesis are in `configs/thesis/`; see
`configs/README.md` for the config schema and override rules.

## Quickstart

Verify the install end-to-end on the bundled miniature corpus:

    bash examples/run.sh

## Fine-tuning on your own data

You can continue training the released checkpoint
([`n-bor1sov/RuModernBERT-base-reranker-ru`](https://huggingface.co/n-bor1sov/RuModernBERT-base-reranker-ru))
on a domain corpus with stage 6 alone. You need one CUDA GPU or more,
with `flash-attn` and `clearml` installed.

**1. Prepare the data.** Put Parquet files under two split folders. Each
file name becomes the dataset name unless you add a `dataset` column:

    my_data/
      train/*.parquet
      validation/*.parquet

| Column                         | Required | Description                                                     |
|--------------------------------|----------|-----------------------------------------------------------------|
| `query`                        | yes      | Query text                                                      |
| `positive`                     | yes      | Relevant passage                                                |
| `negative_1` … `negative_N`    | yes      | Hard negatives (non-relevant but similar passages), `N ≥ k_h`   |
| `query_id`                     | yes      | Stable query id; rows with the same id stay in one bucket       |
| `dataset`                      | no       | Source name; batches never mix datasets                         |
| `lang`                         | no       | ISO 639-1 code, used for per-language validation metrics        |

If you only have `(query, passage)` pairs, mine the negatives with stages
1–5 first (this needs an embedding endpoint, see `docs/PIPELINE.md`). Then
add a `query_id` column to the compiled Parquet.

**2. Bucket the data** (this step runs once):

    python pipeline/06_train/prepare_buckets.py \
      --input_root my_data --output_root work/buckets \
      --num_buckets 64 --num_negatives 8

**3. Train**, starting from the released reranker:

    CLEARML_OFFLINE_MODE=1 torchrun --nproc_per_node=1 pipeline/06_train/train.py \
      --base-model n-bor1sov/RuModernBERT-base-reranker-ru \
      --train-dataset-path work/buckets/train \
      --eval-dataset-path work/buckets/validation \
      --output-dir work/finetuned \
      --max-seq-length 512 \
      --learning-rate 5e-6 --warmup-ratio 0.05 --num-epochs 1 \
      --k-h 8 --k-r 2 \
      --batch-size-per-gpu 64 --mini-batch-size 16 \
      --eval-steps-per-epoch 200

- `--eval-steps-per-epoch` is the interval between evaluations and
  checkpoints, in steps.
- `--mini-batch-size` is the GradCache chunk size. It controls VRAM use
  and does not change the effective batch.
- `--batch-size-per-gpu` sets the contrastive pool. Lower it for small
  datasets: every batch is drawn from a single dataset.
- Do not pass `--config configs/thesis/06_train.toml` here. It trains
  from the raw base model with `max_seq_length = 4096`.

The best checkpoint by validation MAP is reloaded at the end and saved
to `work/finetuned/final_model`. It loads with
`CrossEncoder("work/finetuned/final_model")`.

> **Note:** `train.py` currently hard-codes `HF_HOME` / `HF_DATASETS_CACHE`
> at the top of the file. Point them at a writable directory before you
> run it.

## Reproducing the thesis run

See `docs/REPRODUCING_THESIS.md`. The short version:

    ./run_pipeline.sh
    bash evaluation/run.sh

Each stage can also be run individually:

    bash pipeline/01_split/run.sh
    bash pipeline/02_embed/run.sh
    # ...

For the ablation studies, see `docs/REPRODUCING_ABLATIONS.md`.

## Input data

The pipeline expects a corpus of query–passage pairs already in a
common schema. Expected columns and constraints are documented in
`docs/PIPELINE.md` under "Input data requirements". Helpers for
normalising common source formats are in `data_prep/`.

## License

MIT. See `LICENSE`.
