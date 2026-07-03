# MarketMood
*Investor Sentiment and Price Context for Abnormal Stock-Move Detection*

## Problem Statement

MarketMood studies whether StockTwits-style investor posts, combined with recent (to post) ticker-level price context, can help detect short-term abnormal stock movements. For each post, the system predicts one of three classes for the next trading-day move of the referenced ticker: `positive`, `neutral`, or `negative`, indicating signals of next day abnormal price action.

This is an educational research demo only. It is not financial advice and is not intended for real-money trading decisions.

## Data Sources

The text data come from the StockEmotions dataset, stored locally in `data/stockemo/`. The available splits contain 10,000 posts:

| Split | Rows |
|---|---:|
| Train | 8,000 |
| Validation | 1,000 |
| Test | 1,000 |

The dataset spans January 1, 2020 through December 31, 2020 and covers 37 tickers. The most common tickers are TSLA, AAPL, BA, DIS, and AMZN. Sentiment labels are roughly balanced, with 5,474 bullish posts and 4,526 bearish posts.

Historical daily OHLCV prices are downloaded with `yfinance` and cached locally in `data/prices/{ticker}.csv`. The current cache contains 37 non-empty ticker files. Ticker aliases are used where needed due to symbol changes over time (e.g `BRK.B -> BRK-B` and `FB -> META`).

## Related Work

This project builds on the StockEmotions dataset and paper, which provide emotion and sentiment annotations for stock-market social-media posts. Prior work on financial NLP commonly models sentiment, market reactions, or price movement from social media, news, and price-history signals. 

* [StockEmotions Github](https://github.com/adlnlp/stockemotions)
* [StockEmotions Paper](https://arxiv.org/abs/2301.09279)

MarketMood's contribution on top of StockEmotions combines that approach with technical-analysis rules, classical TF-IDF models, and transformer-plus-price fusion for volatility-normalized abnormal-move detection that can potentially act as buy/sell signals for trading strategies.

[TODO] 2-4 relevant financial NLP or stock-movement prediction papers.

## Evaluation Strategy And Metrics

The primary metric will be macro F1 because the target distribution is imbalanced toward `neutral`, and a model that performs poorly on `positive` or `negative` abnormal moves should be penalized. Secondary metrics will include accuracy, weighted F1, per-class precision/recall/F1, and confusion matrices.

Model selection and hyperparameter tuning will use train and validation data only. The test split will be reserved for final comparison.

Current modeling-dataset class distribution:

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 1,721 | 4,226 | 2,000 | 7,947 |
| Validation | 230 | 515 | 254 | 999 |
| Test | 198 | 539 | 257 | 994 |
| All | 2,149 | 5,280 | 2,511 | 9,940 |

## Data Processing Pipeline

The pipeline loads local StockEmotions train, validation, and test CSVs; parses dates; normalizes ticker symbols; downloads and caches daily OHLCV data; aligns posts to ticker trading dates; constructs leakage-safe price features; builds text input variants; and writes `data/processed/modeling_dataset.csv`.

The event date for a post is the first available ticker trading date on or after the post calendar date. Price features use data through `t-1`. The target uses the future return from `close(t)` to `close(t+1)`.

The abnormal-move score is:

```text
future_return_1d = close(t+1) / close(t) - 1
rolling_vol_20d = standard deviation of daily returns over the prior 20 trading days
abnormal_score = future_return_1d / rolling_vol_20d
```

Labels use threshold `0.75`:

```text
positive if abnormal_score > +0.75
negative if abnormal_score < -0.75
neutral otherwise
```

Generated price features include returns over 1, 3, 5, 10, and 20 days; volatility over 5, 10, and 20 days; 20-day volume z-score; 5- and 20-day moving averages; close-to-SMA20; high-low range; gap return; 20-day range position; and volatility-adjusted breakout/breakdown strength.

## Modeling Approach

The project will compare three required modeling families:

1. Naive baseline: a deterministic technical-analysis rule using 20-day range position, volatility-adjusted breakout/breakdown strength, and 5-day momentum.
2. Classical ML: TF-IDF text features, engineered price features, and logistic regression variants.
3. Deep learning: transformer text encoder with optional price-feature MLP fusion.

The focused experiment will be an ablation study asking whether investor text adds signal beyond price-only baselines, and whether price context improves text-only models.

## Hyperparameter Tuning Strategy

Hyperparameters will be selected using validation macro F1. For classical ML, planned tuning includes TF-IDF vocabulary size, n-gram range, regularization strength, and feature-set choice. For deep learning, planned tuning includes encoder choice, max sequence length, learning rate, batch size, dropout, class weighting, and text-only versus text-plus-price fusion.

No hyperparameter choices will be made using test performance.

## Models Evaluated

| Model | Status | Notes |
|---|---|---|
| Technical-analysis baseline | Complete | Deterministic volatility-adjusted range-breakout rule, no training |
| Classical price-only | Pending | Engineered price features |
| Classical text-only | Pending | TF-IDF over original post text |
| Classical text + price | Pending | TF-IDF plus engineered price features |
| Deep text-only transformer | Pending | Transformer classifier |
| Deep text + price fusion | Pending | Transformer embedding plus price MLP |

## Results

Current quantitative results:

| Model | Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Technical-analysis baseline | Test | 0.476 | 0.412 | 0.475 |

Technical-analysis baseline confusion matrix, ordered as `negative`, `neutral`, `positive`:

| True \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 45 | 89 | 64 |
| Neutral | 42 | 323 | 174 |
| Positive | 38 | 114 | 105 |

The technical-analysis rule is intentionally simple but now mirrors the target more closely than pure trailing momentum. It predicts a positive setup when the prior close sits in the top 20% of its recent 20-day range and either 5-day momentum or 20-day breakout strength exceeds `0.75` trailing-volatility units. It predicts a negative setup symmetrically near the bottom 20% of the recent range. All other rows are neutral. On the test set, this produces a more neutral-aware baseline than the earlier pure-momentum rule.

Current completed outputs:

- `notebooks/01_stockemo_eda.ipynb`
- `notebooks/02_modeling_dataset_eda.ipynb`
- `data/processed/modeling_dataset.csv`
- `data/processed/modeling_dataset_dropped_rows.csv`
- `data/processed/modeling_dataset_class_distribution.csv`
- `outputs/predictions/ta_baseline_test_predictions.csv`
- `outputs/metrics/ta_baseline_metrics.json`

## Error Analysis

Error analysis is pending model predictions. The final report will identify at least five specific mispredictions, discuss likely root causes, and propose concrete mitigations. Planned categories include ambiguous investor language, ticker-specific news events, high-volatility regimes, posts with sarcasm or emojis, and cases where text sentiment conflicts with subsequent price movement.

## Experiment Write-Up

Planned experiment: ablation study of price-only, text-only, and text-plus-price modeling.

Research question: Does investor text add predictive signal beyond recent technical price context, and does price context improve transformer-based prediction?

Planned comparisons:

| Condition | Status |
|---|---|
| Technical-analysis baseline | Complete |
| Classical price-only | Pending |
| Classical text-only | Pending |
| Classical text + price | Pending |
| Deep text-only transformer | Pending |
| Deep text + price fusion transformer | Pending |

Interpretation and recommendations will be added after results are generated.

## Recommendations


## Conclusions


## Future Work

With more time, useful extensions would include:
* more recent market and sentiment data
* backtesting with some simple strategies to see if the signals are tradeable

Additionally:
 richer market-context features, intraday post-time alignment, predicted sentiment/emotion features instead of oracle labels, news-event controls, model calibration, explainability, and backtesting

## Commercial Viability Statement

MarketMood is not currently suitable for commercial trading or investment decisions:
* The dataset is limited to 2020
* post timestamps are date-level rather than intraday, 
* social-media posts are noisy and selection-biased, and abnormal price moves can be driven by external news that is not in the feature set. 

This is for research and educational investor-sentiment dashboard.

## Ethics Statement

The predictions are educational and not financial advice.

## Rubric Alignment Snapshot

| Rubric requirement | Current status |
|---|---|
| Naive baseline | Complete: technical-analysis baseline implemented and evaluated |
| Classical non-deep-learning model | Pending |
| Neural-network deep-learning model | Pending |
| Focused experiment | Planned as a price-only, text-only, and text-plus-price ablation |
| Interactive inference app | Placeholder exists; model-loading app still pending |
| Written report | Draft started and updated with current data pipeline and baseline results |
| Error analysis | Pending model predictions beyond baseline |
| Deployment link | Pending |
