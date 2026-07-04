---
title: AIPI540 MarketMood
sdk: gradio
app_file: app/app.py
python_version: 3.11
suggested_hardware: cpu-basic
startup_duration_timeout: 1h
---

# MarketMood

MarketMood is an educational NLP and machine learning project for detecting short-term abnormal stock moves from StockTwits-style investor posts and recent market price context.

The project will compare:

- A deterministic technical-analysis baseline.
- Classical machine learning with TF-IDF and engineered price features.
- A transformer-based deep learning model with text and price-feature fusion.

This is a research demo only and is not financial advice.

## Setup

Use Python 3.13
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Data

Place StockEmotions CSV files in `data/stockemo/`:

- `train_stockemo.csv`
- `val_stockemo.csv`
- `test_stockemo.csv`

Historical prices will be cached under `data/prices/`.

## Main Commands

```bash
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/evaluate_models.py
python app/app.py
```

`prepare_data.py` caches prices and builds `data/processed/modeling_dataset.csv`.
`train_models.py` trains all implemented trainable models. The current technical-analysis baseline has no training step.
`evaluate_models.py` evaluates all implemented models and writes metrics/predictions under `outputs/`.

## App Demo

The Gradio demo uses the saved deep `text_price` model as the official prediction and shows comparison rows for the other saved models. Select a ticker and event date from the cached historical data, edit or choose a sample investor post, and run the models against the message plus prior price context.

```bash
python app/app.py
```

## Hugging Face Spaces Deployment

The Gradio app can run locally from repo-relative artifacts or in Hugging Face Spaces from a mounted storage bucket. For Spaces, runtime artifacts are expected under:

```text
/data/marketmood/models
/data/marketmood/data/processed
/data/marketmood/data/prices
```

The deploy helper syncs the required runtime artifacts to `hf://buckets/hw391/AIPI540-MarketMood-storage`, configures the bucket mount and runtime variables, and uploads the app/source files. Spaces rebuild automatically after source uploads.

Preview the deployment actions:

```bash
python scripts/deploy_space.py --dry-run
```

Deploy or redeploy:

```bash
python scripts/deploy_space.py
```

For source-only redeploys after UI/code changes:

```bash
python scripts/deploy_space.py --skip-artifacts --skip-volume --skip-variables
```

If the Space needs a manual restart and the CLI can resolve it:

```bash
python scripts/deploy_space.py --skip-artifacts --skip-volume --skip-variables --skip-source --restart --wait
```
