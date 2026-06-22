"""
Predix — Batch Scoring Pipeline
────────────────────────────────────────────────────────────────────────────────
Loads champion models from MLflow, scores the full user base, and writes
output Parquet files consumed by the FastAPI serving layer.

Run daily via Airflow dag_04_batch_scoring.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow.sklearn
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHURN_FEATURE_COLS = [
    "frequency", "monetary", "avg_price", "recency",
    "unique_cats", "unique_brands", "unique_prods",
    "n_views", "n_carts", "cart_to_pur_ratio",
]


def load_champion(model_name: str, stage: str = "Production") -> object:
    log.info("Loading model: %s (%s)", model_name, stage)
    model_uri = f"models:/{model_name}/{stage}"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        log.warning("Model '%s' not in registry. Loading from local mlruns…", model_name)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise RuntimeError(f"No versions found for model '{model_name}'")
        run_id  = versions[-1].run_id
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def score_churn(
    features_path: str,
    output_path:   str,
    model_name:    str = "predix_churn_model",
    stage:         str = "Production",
) -> None:
    feat = pd.read_parquet(features_path)
    model = load_champion(model_name, stage)

    missing = [c for c in CHURN_FEATURE_COLS if c not in feat.columns]
    if missing:
        log.warning("Missing churn feature columns: %s — filling with 0", missing)
        for c in missing:
            feat[c] = 0

    feat["churn_score"]  = model.predict_proba(feat[CHURN_FEATURE_COLS].values)[:, 1]
    feat["churn_risk"]   = pd.cut(
        feat["churn_score"],
        bins=[0, 0.35, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    feat["scored_at"] = pd.Timestamp.utcnow()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "churn_scores.parquet"
    feat[["user_id", "churn_score", "churn_risk", "scored_at"]].to_parquet(out_file, index=False)
    log.info("Churn scores written to %s (%d users).", out_file, len(feat))


def score_upsell(
    user_scores_path: str,
    output_path:      str,
) -> None:
    scores = pd.read_parquet(user_scores_path)
    scores["scored_at"] = pd.Timestamp.utcnow()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "upsell_scores.parquet"
    scores.to_parquet(out_file, index=False)
    log.info("Upsell scores written to %s (%d rows).", out_file, len(scores))


def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.mlflow_uri)

    if args.task in ("churn", "all"):
        score_churn(
            features_path = args.churn_features,
            output_path   = args.output_dir,
            model_name    = args.churn_model,
            stage         = args.stage,
        )

    if args.task in ("upsell", "all"):
        score_upsell(
            user_scores_path = args.upsell_scores,
            output_path      = args.output_dir,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predix — Batch Scoring")
    p.add_argument("--task",            choices=["churn", "upsell", "all"], default="all")
    p.add_argument("--churn-features",  default="./pipeline/scoring/churn_features.parquet")
    p.add_argument("--upsell-scores",   default="./pipeline/scoring/user_scores.parquet")
    p.add_argument("--output-dir",      default="./pipeline/scoring/output")
    p.add_argument("--churn-model",     default="predix_churn_model")
    p.add_argument("--stage",           default="Production")
    p.add_argument("--mlflow-uri",      default="./mlruns")
    main(p.parse_args())
