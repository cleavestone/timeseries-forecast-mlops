
# Time Series Forecasting MLOps Platform

An end-to-end, production-style ML system for retail demand forecasting — built as a
portfolio project to demonstrate both applied time series modeling and real-world
MLOps engineering practices.

## Project Goals

This project is deliberately scoped to cover three things in depth, not just as a checklist:

1. **Time series forecasting with ML** — a proper progression from naive baselines
   through classical statistical models, gradient-boosted trees, and a global deep
   learning model, evaluated with consistent time-respecting backtests.
2. **MLOps best practices** — experiment tracking, data versioning, a model registry
   with a real promotion lifecycle, and CI/CD gated on evaluation metrics, not just
   passing tests.
3. **Production monitoring** — live metrics, drift detection, alerting, and a
   drift-triggered retraining loop, not just a dashboard for show.

## Status

🚧 Actively in development (started July 2026). See [Roadmap](#roadmap) below.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) — daily sales
across ~1,000 stores over ~3 years, including promotions, holidays, and store
closures. Chosen for its balance of real-world messiness and manageable scale.

## Architecture

*(Diagram to be added once the pipeline is built end-to-end.)*

High-level flow: raw data → DVC-tracked feature pipeline → model training with
MLflow tracking → best model promoted to MLflow Model Registry → FastAPI batch
prediction service → predictions logged to Postgres → Prometheus/Grafana monitoring
→ Kestra-orchestrated scheduled inference, drift checks, and conditional retraining
→ GitHub Actions CI/CD gating merges on evaluation metrics.

## Tech Stack

| Concern                        | Tool                                           |
| ------------------------------ | ---------------------------------------------- |
| Modeling                       | LightGBM, SARIMA, N-BEATS/TFT (neuralforecast) |
| Hyperparameter tuning          | Optuna                                         |
| Experiment tracking & registry | MLflow                                         |
| Data versioning                | DVC (MinIO remote)                             |
| Orchestration                  | Kestra                                         |
| Serving                        | FastAPI (batch prediction)                     |
| Storage                        | PostgreSQL                                     |
| Drift detection                | Evidently AI                                   |
| Metrics & dashboards           | Prometheus + Grafana                           |
| CI/CD                          | GitHub Actions                                 |
| Package management             | uv                                             |
| Infrastructure                 | Docker + Docker Compose                        |

## Roadmap

- [ ] Data ingestion, EDA, time-respecting train/val/test split
- [ ] Feature engineering pipeline (shared between training and inference)
- [ ] Baseline models (seasonal naive, SARIMA)
- [ ] LightGBM model + MLflow tracking
- [ ] Optuna hyperparameter tuning
- [ ] Global deep learning model (N-BEATS/TFT) comparison
- [ ] Model registry promotion workflow
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
