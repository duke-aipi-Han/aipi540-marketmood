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

The project compares:

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

Download StockEmotions CSV files from: `https://github.com/adlnlp/StockEmotions/` and then place CSV files in `data/stockemo/`:

- `train_stockemo.csv`
- `val_stockemo.csv`
- `test_stockemo.csv`

Historical prices will be cached under `data/prices/`.

## Main Commands

```bash
python scripts/prepare_data.py # download price, generate features, prepare data for ML
python scripts/train_models.py # train the various comparison models
python scripts/evaluate_models.py # run test on generated models
python scripts/run_backtest.py # run simulated backtest using simple strategy
python scripts/generate_report_artifacts.py # generate data and charts needed for report
python app/app.py # start Gradio app locally
python scripts/deploy_space.py # deploys Gradio app and necessary artifacts to Huggingface Spaces

```

`prepare_data.py` caches prices and builds `data/processed/modeling_dataset.csv`.
`train_models.py` trains all implemented trainable models. The current technical-analysis baseline has no training step.
`evaluate_models.py` evaluates all implemented models and writes metrics/predictions under `outputs/`.
`run_backtest.py` runs a simple one-day simulated trading backtest from saved prediction CSVs.
`generate_report_artifacts.py` converts saved metrics and predictions into report-ready figures and error-analysis tables.

## Models And Source Locations

| Model | Purpose | Training source | Inference/evaluation source | Saved artifacts |
|---|---|---|---|---|
| Technical-analysis baseline | Deterministic price-action baseline | No training; rule in `src/marketmood/baselines/technical_baseline.py` | `src/marketmood/baselines/technical_baseline.py`, `scripts/evaluate_models.py` | `outputs/predictions/ta_baseline_test_predictions.csv`, `outputs/metrics/ta_baseline_metrics.json` |
| Classical price-only | Logistic regression over engineered price features | `src/marketmood/training/train_classical.py`, `src/marketmood/models/classical.py` | `src/marketmood/models/classical.py`, `src/marketmood/inference/predict.py` | `models/classical/price_only.joblib` |
| Classical text-only | TF-IDF post text plus logistic regression | `src/marketmood/training/train_classical.py`, `src/marketmood/models/classical.py` | `src/marketmood/models/classical.py`, `src/marketmood/inference/predict.py` | `models/classical/text_only.joblib` |
| Classical text + price | TF-IDF post text fused with engineered price features | `src/marketmood/training/train_classical.py`, `src/marketmood/models/classical.py` | `src/marketmood/models/classical.py`, `src/marketmood/inference/predict.py` | `models/classical/text_price.joblib` |
| Deep text-only | Frozen DistilBERT text encoder plus classifier head | `src/marketmood/training/train_deep_fusion.py`, `src/marketmood/models/deep_fusion.py` | `src/marketmood/models/deep_fusion.py`, `src/marketmood/inference/predict.py` | `models/deep_fusion/text_only/model.pt` |
| Deep text + price | Frozen DistilBERT text encoder fused with price-feature MLP | `src/marketmood/training/train_deep_fusion.py`, `src/marketmood/models/deep_fusion.py` | `src/marketmood/models/deep_fusion.py`, `src/marketmood/inference/predict.py` | `models/deep_fusion/text_price/model.pt` |

The Gradio app in `app/app.py` uses `src/marketmood/inference/predict.py` to load saved artifacts. The official app prediction is the deep text + price model, with other saved models shown for comparison.

## Repository Layout

| Path | Description |
|---|---|
| `app/app.py` | Gradio inference dashboard used locally and on Hugging Face Spaces |
| `config.yaml` | Central configuration for paths, features, labels, baseline settings, and model settings |
| `data/stockemo/` | Local StockEmotions CSV inputs |
| `data/prices/` | Cached yfinance OHLCV price histories |
| `data/processed/` | Generated modeling dataset and processing summaries |
| `models/classical/` | Saved classical ML model artifacts |
| `models/deep_fusion/` | Saved deep-learning model artifacts and tokenizers |
| `notebooks/` | Exploratory and model-analysis notebooks |
| `outputs/backtest/` | Simulated signal-backtest trades, equity curves, and summary metrics |
| `outputs/error_analysis/` | Generated misclassification and model-disagreement examples |
| `outputs/figures/` | Report-ready plots, confusion matrices, and backtest figures |
| `outputs/metrics/` | Evaluation metrics, model comparison tables, and subgroup metrics |
| `outputs/predictions/` | Saved test-set prediction CSVs by model |
| `scripts/` | Command-line wrappers for data prep, training, evaluation, backtesting, report artifacts, and deployment |
| `src/marketmood/` | Main Python package for data loading, features, labels, models, training, evaluation, inference, and backtesting |
| `tests/` | Pytest coverage for labels, features, models, inference, report artifacts, and backtesting |
| `report.md` | Technical report with metrics, figures, error analysis, and backtest results |

## Simulated Backtest

The backtest is an educational signal simulation, not a trading system. It starts with `$1,000,000`, aggregates duplicate posts into one signal per model/ticker/event date, scales position size by model confidence, goes long on `positive`, short on `negative`, and holds cash on `neutral`. Each position uses the same one-day return window as the target: `close(t)` to `close(t+1)`.

```bash
python scripts/run_backtest.py
```

Optional settings:

```bash
python scripts/run_backtest.py --initial-capital 1000000 --max-position-pct 0.05 --max-daily-gross-exposure 1.0
```

Outputs are written to `outputs/backtest/` and backtest plots are written to `outputs/figures/`.

## App

The Gradio app uses the saved deep `text_price` model as the official prediction and shows comparison rows for the other saved models. Select a ticker and event date from the cached historical data, edit or choose a sample investor post, and run the models against the message plus prior price context.

Deployed demo: https://hw391-aipi540-marketmood.hf.space

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

The deploy helper syncs the required runtime artifacts to `hf://buckets/hw391/AIPI540-MarketMood-storage`, configures the bucket mount and runtime variables, and uploads the app/source files plus the small prediction CSVs used by the Interesting examples dropdown. Spaces rebuild automatically after source uploads.

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
