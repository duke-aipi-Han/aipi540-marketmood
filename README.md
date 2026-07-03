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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
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
