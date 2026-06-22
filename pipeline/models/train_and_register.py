"""
Predix — Training Entrypoint
────────────────────────────────────────────────────────────────────────────────
Orchestrates the full training loop:
  1. Load data from Gold layer (Snowflake / local Parquet)
  2. Build churn features
  3. Train and evaluate all classifiers
  4. Register champion in MLflow Model Registry
  5. Train upsell model and export scores

Usage:
    python pipeline/models/train_and_register.py \
        --data-dir ./dataset \
        --mlflow-uri ./mlruns \
        --sample-rows 2000000
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.models.churn_model  import ChurnModel, ChurnModelConfig
from pipeline.models.upsell_model import UpsellModel, UpsellModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_data(data_dir: Path, sample_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading data from %s (sample=%d per file)…", data_dir, sample_rows)
    oct_df = pd.read_csv(data_dir / "2019-Oct.csv", nrows=sample_rows)
    nov_df = pd.read_csv(data_dir / "2019-Nov.csv", nrows=sample_rows)

    for df in [oct_df, nov_df]:
        df["event_time"]    = pd.to_datetime(df["event_time"], utc=True)
        df["price"]         = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["brand"]         = df["brand"].fillna("unknown")
        df["category_code"] = df["category_code"].fillna("")
        df["category"]      = df["category_code"].str.split(".").str[0].replace("", "unknown")

    log.info("Oct: %d rows | Nov: %d rows", len(oct_df), len(nov_df))
    return oct_df, nov_df


def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.mlflow_uri)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    oct_df, nov_df = load_data(data_dir, args.sample_rows)
    full_df = pd.concat([oct_df, nov_df], ignore_index=True)

    # ── Churn model ───────────────────────────────────────────────────────────
    log.info("Starting churn model training…")
    churn_cfg = ChurnModelConfig(
        experiment_name  = "predix_churn_v1",
        gb_n_estimators  = 150,
        gb_max_depth     = 4,
        gb_learning_rate = 0.05,
    )
    churn_model = ChurnModel(cfg=churn_cfg)
    feat = ChurnModel.build_features(oct_df, nov_df)
    churn_model.fit(feat)

    log.info("\n%s", churn_model.summary().to_string(index=False))

    run_id = churn_model.register_champion()
    log.info("Champion registered. MLflow run_id=%s", run_id)

    # ── RFM (needed for upsell) ───────────────────────────────────────────────
    from pipeline.feature_store.feature_engineering import build_rfm
    rfm = build_rfm(full_df)

    # ── Upsell model ──────────────────────────────────────────────────────────
    log.info("Starting upsell model training…")
    upsell_cfg = UpsellModelConfig(experiment_name="predix_upsell_v1")
    upsell_model = UpsellModel(cfg=upsell_cfg)
    upsell_model.fit(full_df, rfm)

    scores_path = ROOT / "pipeline" / "scoring" / "user_scores.parquet"
    upsell_model.export_scores(str(scores_path))
    log.info("Upsell scores written to %s", scores_path)

    # ── Churn scores ─────────────────────────────────────────────────────────
    feat["churn_score"] = churn_model.predict_proba(feat)
    churn_path = ROOT / "pipeline" / "scoring" / "churn_scores.parquet"
    feat[["user_id", "churn_score"]].to_parquet(churn_path, index=False)
    log.info("Churn scores written to %s", churn_path)

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predix — Train & Register models")
    parser.add_argument("--data-dir",    default="./dataset",  type=str)
    parser.add_argument("--mlflow-uri",  default="./mlruns",   type=str)
    parser.add_argument("--sample-rows", default=2_000_000,    type=int)
    main(parser.parse_args())
