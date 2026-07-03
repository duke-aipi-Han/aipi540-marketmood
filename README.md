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

## Planned Commands

```bash
python -m marketmood.prices --config config.yaml
python -m marketmood.features --config config.yaml
python -m marketmood.training.train_classical --config config.yaml
python -m marketmood.training.train_deep_fusion --config config.yaml
python -m marketmood.evaluation.evaluate_models --config config.yaml
python app/app.py
```
