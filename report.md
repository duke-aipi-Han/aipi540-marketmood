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

The focused experiment is an ablation study asking whether investor text adds signal beyond price-only baselines, and whether price context improves text-only models.

## Hyperparameter Tuning Strategy

Hyperparameters will be selected using validation macro F1. For classical ML, planned tuning includes TF-IDF vocabulary size, n-gram range, regularization strength, and feature-set choice. For deep learning, planned tuning includes encoder choice, max sequence length, learning rate, batch size, dropout, class weighting, and text-only versus text-plus-price fusion.

No hyperparameter choices will be made using test performance.

## Models Evaluated

| Model | Status | Notes |
|---|---|---|
| Technical-analysis baseline | Complete | Deterministic volatility-adjusted range-breakout rule, no training |
| Classical price-only | Complete | Logistic regression over engineered price features |
| Classical text-only | Complete | TF-IDF over original post text, logistic regression |
| Classical text + price | Complete | TF-IDF plus engineered price features, logistic regression |
| Deep text-only transformer | Complete | Frozen DistilBERT embeddings plus classifier head |
| Deep text + price fusion | Complete | Frozen DistilBERT embeddings plus engineered-price MLP fusion |

## Results

Current quantitative results:

| Model | Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Deep text + price | Test | 0.652 | 0.624 | 0.653 |
| Classical price-only | Test | 0.497 | 0.464 | 0.511 |
| Classical text + price | Test | 0.504 | 0.457 | 0.512 |
| Technical-analysis baseline | Test | 0.476 | 0.412 | 0.475 |
| Classical text-only | Test | 0.416 | 0.365 | 0.424 |
| Deep text-only | Test | 0.385 | 0.328 | 0.386 |

Validation results used for model selection:

| Model | Validation Accuracy | Validation Macro F1 | Validation Weighted F1 |
|---|---:|---:|---:|
| Classical price-only | 0.481 | 0.458 | 0.494 |
| Classical text + price | 0.482 | 0.450 | 0.489 |
| Classical text-only | 0.389 | 0.349 | 0.393 |
| Deep text + price | 0.663 | 0.644 | 0.664 |
| Deep text-only | 0.363 | 0.316 | 0.357 |

Deep text-plus-price confusion matrix on the test split, ordered as `negative`, `neutral`, `positive`:

| True \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 125 | 47 | 26 |
| Neutral | 55 | 385 | 99 |
| Positive | 28 | 91 | 138 |

Technical-analysis baseline confusion matrix, ordered as `negative`, `neutral`, `positive`:

| True \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 45 | 89 | 64 |
| Neutral | 42 | 323 | 174 |
| Positive | 38 | 114 | 105 |

The technical-analysis rule is intentionally simple but now mirrors the target more closely than pure trailing momentum. It predicts a positive setup when the prior close sits in the top 20% of its recent 20-day range and either 5-day momentum or 20-day breakout strength exceeds `0.75` trailing-volatility units. It predicts a negative setup symmetrically near the bottom 20% of the recent range. All other rows are neutral. On the test set, this produces a more neutral-aware baseline than the earlier pure-momentum rule.

The classical logistic-regression models use only deployable features based on text and engineered price features. They do not use `senti_label` or `emo_label` as model inputs as that would be leakage. The current best classical model by validation macro F1 is price-only logistic regression. Text-only TF-IDF performs worse, suggesting that the StockTwits post text alone is not enough for this target, though the text+price model remains close to price-only on weighted F1 and accuracy.

The deep-learning models use DistilBERT text embeddings with a classifier head. To keep training practical on the local environment, the text encoder is frozen by default and only the classification/fusion layers are trained; this can be changed in `config.yaml` by setting `freeze_text_encoder: false`. The latest 8-epoch run used Apple MPS and selected epoch 7 by validation macro F1. The deep text-only model remains weak, while the deep text-plus-price fusion model performs best overall, improving test macro F1 to 0.624 and showing that transformer text representations become useful when combined with leakage-safe market context.

Current completed outputs:

- `notebooks/01_stockemo_eda.ipynb`
- `notebooks/02_modeling_dataset_eda.ipynb`
- `data/processed/modeling_dataset.csv`
- `data/processed/modeling_dataset_dropped_rows.csv`
- `data/processed/modeling_dataset_class_distribution.csv`
- `outputs/predictions/ta_baseline_test_predictions.csv`
- `outputs/metrics/ta_baseline_metrics.json`
- `models/classical/price_only.joblib`
- `models/classical/text_only.joblib`
- `models/classical/text_price.joblib`
- `models/deep_fusion/text_only/model.pt`
- `models/deep_fusion/text_price/model.pt`
- `outputs/predictions/classical_price_only_test_predictions.csv`
- `outputs/predictions/classical_text_only_test_predictions.csv`
- `outputs/predictions/classical_text_price_test_predictions.csv`
- `outputs/predictions/deep_text_only_test_predictions.csv`
- `outputs/predictions/deep_text_price_test_predictions.csv`
- `outputs/metrics/classical_validation_metrics.json`
- `outputs/metrics/classical_metrics.json`
- `outputs/metrics/deep_fusion_validation_metrics.json`
- `outputs/metrics/deep_fusion_metrics.json`
- `outputs/metrics/experiment_summary.csv`

## Error Analysis

Error analysis notebooks now include representative misclassification examples for the technical-analysis baseline and classical models. The final report will identify at least five specific mispredictions, discuss likely root causes, and propose concrete mitigations. Planned categories include ambiguous investor language, ticker-specific news events, high-volatility regimes, posts with sarcasm or emojis, and cases where text sentiment conflicts with subsequent price movement.

## Experiment Write-Up

Experiment: ablation study of price-only, text-only, and text-plus-price modeling.

Research question: Does investor text add predictive signal beyond recent technical price context, and does price context improve transformer-based prediction?

Completed comparisons:

| Condition | Status |
|---|---|
| Technical-analysis baseline | Complete |
| Classical price-only | Complete |
| Classical text-only | Complete |
| Classical text + price | Complete |
| Deep text-only transformer | Complete |
| Deep text + price fusion transformer | Complete |

Current interpretation: engineered price context explains more of the abnormal-return target than sparse text alone. Classical TF-IDF text-only and deep text-only both underperform on macro F1, but deep text-plus-price fusion now beats the deterministic baseline and all classical variants. This supports the main hypothesis that investor text is most useful when interpreted alongside recent market context.

## Recommendations

Use the deep text-plus-price model as the strongest current research model. Keep the technical-analysis baseline, classical price-only model, and classical text-plus-price model as benchmarks because they clarify how much value comes from price context, sparse text features, and transformer-based fusion.


## Conclusions

The current results suggest that the abnormal-move target is difficult to predict from post text alone, but recent price context and transformer-based text representations together improve performance. The best current model is deep text-plus-price fusion with test macro F1 of 0.624.


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
| Classical non-deep-learning model | Complete: price-only, text-only, and text-plus-price logistic regression |
| Neural-network deep-learning model | Complete: DistilBERT text-only and text-plus-price fusion models |
| Focused experiment | Complete: price-only, text-only, and text-plus-price ablation across baseline, classical, and deep models |
| Interactive inference app | Complete: Gradio inference demo uses the saved deep text-plus-price model as official prediction and compares saved model outputs |
| Written report | Draft started and updated with current data pipeline, model results, and interpretation |
| Error analysis | Partially complete in analysis notebooks; final narrative examples still pending |
| Deployment link | Pending |
