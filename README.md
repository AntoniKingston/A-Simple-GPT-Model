# A-Simple-GPT-Model

A small, from-scratch(ish) GPT-style language model implemented in PyTorch, built as a learning project.

Initially, the plan was to train on a corpus consisting solely of **Adam Mickiewicz** works. In practice, that corpus turned out to be **too small** to train a useful model, so the project is still **work in progress**. The next step is to train models on a much larger dataset: the **Polish Parliamentary Corpus (PPC)** from CLARIN-PL.

## What’s in this repo

- **Model**: `SimpleGPT` — a Transformer decoder stack with sinusoidal positional embeddings.
- **Tokenizer**: a very simple **BPE** tokenizer (`tokenizer.py`) trained on your corpus.
- **Training loop**: `train.py` (used from notebooks).
- **Dataset**: `StrideDataset` in `datasets.py` for next-token prediction with a sliding window.
- **Utilities**: save/load model, tokenizer, and encodings in `helpers.py`.
- **Notebooks**:
  - `training.ipynb` / `training_010.ipynb` / `training_020.ipynb` — training experiments
  - `generate.ipynb` — text generation from a saved model + tokenizer

## Status

- **In progress**: the code works for small experiments and Mickiewicz-based prototypes, but the overall project goal is to move to a much larger corpus (PPC) and iterate on training quality, evaluation, and generation.
- Expect some rough edges (paths, naming conventions, notebook-driven workflow).

## Data sources

### Adam Mickiewicz corpus (prototype)

The script `create_mickiewicz_corpus.py` downloads public-domain texts from Wolne Lektury and merges them into a single corpus with `<BOS>`/`<EOS>` markers.

### Planned: Polish Parliamentary Corpus (PPC)

Target dataset for the next iteration: **PPC** — [Polish Parliamentary Corpus (CLARIN-PL)](https://clip.ipipan.waw.pl/PPC).

The PPC integration is **not implemented yet** in this repo; it’s the planned direction because it’s orders of magnitude larger than the initial Mickiewicz-only corpus.

## Quickstart

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (Optional) Build the Mickiewicz prototype corpus

```bash
python create_mickiewicz_corpus.py
```

By default it writes the merged corpus under `corpora/` (created automatically).

### 3) Train a tokenizer and create encodings

`encoding_prep.py`:
- trains a BPE tokenizer on the corpus text
- saves the tokenizer checkpoint (vocab + merges)
- encodes the full corpus into token IDs
- splits into train/val encodings and saves them as `.pt`

Example (uses defaults inside the script):

```bash
python encoding_prep.py
```

This will create (or reuse) output directories such as:

- `tokenizers/`
- `encodings/`

Note: `encoding_prep.py` saves files with a **version suffix** (e.g. `tokenizer_mickiewicz_010.pt`, `tr_encoding_mickiewicz_010.pt`). Some notebooks load filenames **without** the version suffix. Adjust either the notebook paths or the `version` argument in `encoding_prep.py` to match what you have on disk.

### 4) Train the model (notebook workflow)

Open one of the training notebooks and run cells top-to-bottom:

- `training.ipynb` (main baseline)
- `training_010.ipynb`, `training_020.ipynb` (experiment snapshots)

They typically:
- load tokenizer from `tokenizers/`
- load train/val encodings from `encodings/`
- build `StrideDataset` with a context size (block size)
- construct `SimpleGPT` with `BlockConfig`
- run `train(...)`
- save a model checkpoint to `models/*.pt`

## Text generation

Use `generate.ipynb` as the reference workflow. At a high level:

1. load tokenizer via `helpers.load_bpe_tokenization(...)`
2. load model via `helpers.load_model(...)`
3. run `model.generate(...)`
4. decode tokens back to text

## Project layout (high-level)

- `simpleGPT.py` — the model + `generate(...)`
- `block.py` — Transformer block + attention/FFN config
- `tokenizer.py` — BPE tokenizer training + encode/decode
- `encoding_prep.py` — corpus → tokenizer + train/val encodings
- `datasets.py` — `StrideDataset`
- `train.py` — training loop and `TrainingConfig`
- `helpers.py` — saving/loading checkpoints
- (runtime-created) `corpora/`, `tokenizers/`, `encodings/`, `models/`


