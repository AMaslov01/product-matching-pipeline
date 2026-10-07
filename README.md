# Product Matching Pipeline

End-to-end product matching built for the Ozon E-CUP 2026 competition. The pipeline combines field-aware text serialization, transformer cross-encoders, typed pair features, and CatBoost fusion.

```text
product records -> canonicalization -> aligned pair text -> cross-encoder score
                              \-----> typed pair features -> CatBoost fusion -> match score
```

## What is included

- Russian text, identifier, unit, color, and attribute normalization
- component-disjoint folds that prevent the same product from leaking across train and validation
- weakly supervised pretraining on LLM-labelled Silver pairs
- Human Gold fine-tuning, hard-negative mining, and out-of-fold evaluation
- transformer scores fused with deterministic pair and catalogue features
- packaging and inference checks that keep training and serving schemas aligned

The main research constraint was evaluation leakage. Human-labelled pairs are connected through shared product IDs, so a random row split can place the same item on both sides of validation. This repository builds folds over connected components instead and uses the resulting out-of-fold predictions for model comparison and fusion.

## Repository scope

This is a data-free portfolio release of the core pipeline. Competition datasets, trained weights, private experiment artifacts, and cloud-operation scripts are intentionally excluded. The code is taken from a fixed committed research snapshot; unfinished local experiments remain in a separate private archive.

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --locked --extra dev --extra local
source .venv/bin/activate

export MATCHCUP_RAW_DIR=/path/to/competition-data
export MATCHCUP_WORK_DIR=/path/to/generated-artifacts

matchcup --config configs/default.yaml doctor
matchcup --config configs/default.yaml canonicalize --dataset human
matchcup --config configs/default.yaml make-folds
matchcup --config configs/default.yaml prepare-pairs --dataset gold
```

Expected raw files are `items.parquet`, `items_human.parquet`, `matches.parquet`, and `matches_llm.parquet`. They are not redistributed here.

Run the data-free checks with:

```bash
uv run --extra dev --extra local pytest
uv run --extra dev ruff check src tests
```

## Evaluation vocabulary

- **Human Gold:** human-labelled product pairs; the only model-selection dataset.
- **Silver:** LLM-labelled pairs used for weak supervision, not final model selection.
- **OOF recipe:** a checkpoint trained without one component-disjoint Gold fold and evaluated only on that omitted fold.
- **Final-like recipe:** a research reproduction of the final schedule, kept separate from OOF model selection.

## License

MIT. The competition data remains subject to the organizer's terms.
