
# Time Series Forecasting MLOps Platform

An end-to-end, production-style ML system for retail demand forecasting — built as a
portfolio project to demonstrate both applied time series modeling and real-world
MLOps engineering practices.

## Project Goals

This project is deliberately scoped to cover three things in depth, not just as a checklist:

1. **Time series forecasting with ML** — a proper progression from naive baselines
   through gradient-boosted trees (with systematic hyperparameter tuning) and a global
   deep learning model, evaluated with consistent time-respecting backtests.
2. **MLOps best practices** — experiment tracking, data versioning, a model registry
   with a real promotion lifecycle, and CI/CD gated on evaluation metrics, not just
   passing tests.
3. **Production monitoring** — live metrics, drift detection, alerting, and a
   drift-triggered retraining loop, not just a dashboard for show.

## Status

🚧 Actively in development (started July 2026). Modeling phase complete; MLOps
infrastructure (serving, monitoring, orchestration, CI/CD) in progress. See
[Roadmap](#roadmap) below.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) — daily sales
across 1,115 stores from 2013-01-01 to 2015-07-31, including promotions, holidays,
and store closures. Chosen for its balance of real-world messiness (missing data,
structural closures, dtype inconsistencies) and manageable scale for rapid iteration.

## Modeling Results

All experiments tracked in MLflow, evaluated on a consistent time-respecting split
(train: 2013-01-01 to 2015-04-30, validation: 2015-05-01 to 2015-06-15, test:
2015-06-16 to 2015-07-31).

| Model                                        | Val WMAPE       | Val RMSE      | Val MAE       |
| -------------------------------------------- | --------------- | ------------- | ------------- |
| Seasonal naive (lag-7 baseline)              | 31.96%          | 3,507         | 2,354         |
| LightGBM (default params)                    | 8.82%           | 926           | 650           |
| **LightGBM (Optuna-tuned, 25 trials)** | **8.30%** | **861** | **611** |
| N-BEATS (global, no exogenous features)      | 18.35%          | 1,874         | 1,351         |

**LightGBM (tuned) is the production model**, registered in the MLflow Model
Registry as `rossmann-sales-forecaster`. Test-set WMAPE (8.31%) closely tracks
validation, indicating the model generalizes rather than overfitting to the
validation window.

**Why N-BEATS underperforms LightGBM here**: this N-BEATS configuration is a pure
autoregressive global model — it sees only each store's historical sales sequence,
with no access to `Promo`, holiday flags, or store metadata. Feature importance
analysis on the LightGBM model shows `Promo` alone ranks among the top few
predictive features, so denying that signal to N-BEATS is a material handicap, not
a fair architecture-vs-architecture comparison. This result is consistent with
a broader, well-documented pattern in forecasting literature: gradient-boosted
trees with well-engineered features frequently outperform deep learning on
small-to-medium structured/tabular time series problems. Extending this to
N-BEATSx or a Temporal Fusion Transformer (which natively support exogenous
features) is noted as future work rather than pursued here, to keep project scope
focused on the MLOps depth this project is primarily demonstrating.

**SARIMA** was deliberately excluded from the model ladder. SARIMA fits one model
per series, which doesn't fit this project's global-model approach across 1,115
stores without a substantially different (and out-of-scope) per-store training/
serving architecture. This tradeoff is a design decision, not an oversight.

## Key Engineering Decisions & Lessons Learned

A few non-obvious issues surfaced during EDA and were fixed before they reached the
training pipeline — documented here since catching these is a meaningful part of
what this project demonstrates:

- **Calendar-aware feature computation**: Naively computing lag/rolling features on
  `Open==1`-filtered rows silently corrupts the weekly seasonality signal for stores
  that close on Sundays (confirmed via ACF comparison between two stores with
  different closure patterns). Fix: every store's series is reindexed to a full
  continuous calendar date range *before* any lag/rolling feature is computed;
  filtering to open days only happens afterward, for training purposes.
- **181 stores have an identical 184-day closure gap** (2014-07-01 to 2014-12-31),
  consistent with real store renovations. Falls entirely within the training
  window, so it doesn't compromise validation/test integrity, but required
  explicit handling in the reindexing logic.
- **`StateHoliday` dtype inconsistency**: the raw column mixes integer `0` and
  string `'0'` as effectively duplicate categories — a silent data quality bug
  that would have fragmented the holiday signal if left uncleaned.
- **Target transform**: trained on `log1p(Sales)`, inverted via `expm1` at
  inference time — reduces right-skew (raw skew 1.59 → -0.64 after transform),
  prevents negative predictions, and treats proportional swings consistently
  across low- and high-volume stores.
- **Evaluation metric**: WMAPE chosen over plain MAPE, since a small number of
  open-day, zero-sales rows exist in the data and would destabilize a plain MAPE
  calculation.
- **Train/serve parity**: all feature engineering logic lives in a single shared
  module (`src/features/build_features.py`), used identically by training and (soon)
  the inference service — avoiding the classic failure mode where training and
  serving silently compute features differently.

## Architecture

*(Diagram to be added once the pipeline is built end-to-end.)*

High-level flow: raw data → DVC-tracked feature pipeline → model training with
MLflow tracking → best model promoted to MLflow Model Registry → FastAPI batch
prediction service → predictions logged to Postgres → Prometheus/Grafana monitoring
→ Kestra-orchestrated scheduled inference, drift checks, and conditional retraining
→ GitHub Actions CI/CD gating merges on evaluation metrics.

## Tech Stack

| Concern                        | Tool                                       |
| ------------------------------ | ------------------------------------------ |
| Modeling                       | LightGBM, N-BEATS (neuralforecast)         |
| Hyperparameter tuning          | Optuna                                     |
| Experiment tracking & registry | MLflow (Postgres backend, MinIO artifacts) |
| Data versioning                | DVC (MinIO remote)                         |
| Orchestration                  | Kestra                                     |
| Serving                        | FastAPI (batch prediction)                 |
| Storage                        | PostgreSQL                                 |
| Drift detection                | Evidently AI                               |
| Metrics & dashboards           | Prometheus + Grafana                       |
| CI/CD                          | GitHub Actions                             |
| Package management             | uv                                         |
| Infrastructure                 | Docker + Docker Compose                    |

## Roadmap

- [X] Data ingestion, EDA, time-respecting train/val/test split
- [X] Feature engineering pipeline (shared between training and inference)
- [X] Baseline model (seasonal naive)
- [X] LightGBM model + MLflow tracking
- [X] Optuna hyperparameter tuning
- [X] Global deep learning model (N-BEATS) comparison
- [X] Model registry promotion workflow
- [ ] FastAPI batch prediction service
- [ ] Prometheus + Grafana monitoring
- [ ] Evidently drift detection
- [ ] Kestra flows: scheduled inference, drift checks, conditional retraining
- [ ] GitHub Actions CI/CD with metric-gated releases
- [ ] Final documentation, architecture diagram, demo

## Why This Project

Built to demonstrate the gap between "training a model in a notebook" and "operating
a model in production" — the modeling is intentionally straightforward so that the
engineering around it (versioning, tracking, serving, monitoring, and the feedback
loop between them) can be the real focus.
