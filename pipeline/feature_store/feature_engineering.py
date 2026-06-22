"""
Predix — Feature Engineering
────────────────────────────────────────────────────────────────────────────────
Builds the analytical feature store from Silver-layer events.
Entities: user, product, session.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REF_DATE = pd.Timestamp("2019-12-01", tz="UTC")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── RFM ───────────────────────────────────────────────────────────────────────
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary + segment for all buyers."""
    pur = df[df["event_type"] == "purchase"].copy()

    rfm = (
        pur.groupby("user_id")
        .agg(
            recency   = ("event_time", lambda x: int((REF_DATE - x.max()).days)),
            frequency = ("event_time", "count"),
            monetary  = ("price", "sum"),
            avg_price = ("price", "mean"),
        )
        .reset_index()
    )
    rfm = rfm[rfm["monetary"] > 0]

    for col, labels, _ in [
        ("recency",   [5, 4, 3, 2, 1], True),
        ("frequency", [1, 2, 3, 4, 5], False),
        ("monetary",  [1, 2, 3, 4, 5], False),
    ]:
        rfm[f"{col[0]}_score"] = pd.qcut(
            rfm[col], q=5, labels=labels, duplicates="drop"
        ).astype(int)

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["segment"]   = rfm.apply(_classify_segment, axis=1)
    log.info("RFM built: %d users | segments: %s", len(rfm), rfm["segment"].value_counts().to_dict())
    return rfm


def _classify_segment(row: pd.Series) -> str:
    r, f = int(row["r_score"]), int(row["f_score"])
    if r >= 4 and f >= 4:  return "Champions"
    if r >= 3 and f >= 3:  return "Loyal"
    if r >= 3 and f <= 2:  return "Potential Loyalist"
    if r == 2 and f >= 3:  return "At Risk"
    if r == 2 and f == 2:  return "Needs Attention"
    if r == 1 and f >= 3:  return "Can't Lose"
    if r == 1 and f == 2:  return "Hibernating"
    return "Lost"


# ── User features ──────────────────────────────────────────────────────────────
def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all event types into a per-user feature table."""
    pur = df[df["event_type"] == "purchase"]
    viw = df[df["event_type"] == "view"]
    crt = df[df["event_type"] == "cart"]

    base = (
        pur.groupby("user_id")
        .agg(
            total_purchases   = ("event_time", "count"),
            total_spend       = ("price", "sum"),
            avg_purchase_price= ("price", "mean"),
            unique_products   = ("product_id", "nunique"),
            unique_categories = ("category_code", "nunique"),
            unique_brands     = ("brand", "nunique"),
            first_purchase_dt = ("event_time", "min"),
            last_purchase_dt  = ("event_time", "max"),
        )
        .reset_index()
    )
    base = base.merge(viw.groupby("user_id").size().rename("total_views"),  on="user_id", how="left")
    base = base.merge(crt.groupby("user_id").size().rename("total_carts"),  on="user_id", how="left")
    base = base.fillna(0)

    base["recency_days"]       = (REF_DATE - base["last_purchase_dt"]).dt.days.astype(int)
    base["cart_to_pur_ratio"]  = base["total_carts"] / base["total_purchases"].replace(0, np.nan)
    base["view_to_pur_ratio"]  = base["total_views"] / base["total_purchases"].replace(0, np.nan)
    base = base.fillna(0)

    log.info("User features built: %d users.", len(base))
    return base


# ── Product features ───────────────────────────────────────────────────────────
def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    pur = df[df["event_type"] == "purchase"]
    viw = df[df["event_type"] == "view"]
    crt = df[df["event_type"] == "cart"]

    prod = (
        pur.groupby(["product_id", "category_code", "brand"])
        .agg(
            total_purchases = ("price", "count"),
            total_revenue   = ("price", "sum"),
            avg_price       = ("price", "mean"),
        )
        .reset_index()
    )
    prod = prod.merge(viw.groupby("product_id").size().rename("total_views"),  on="product_id", how="left")
    prod = prod.merge(crt.groupby("product_id").size().rename("total_carts"),  on="product_id", how="left")
    prod = prod.fillna(0)
    prod["conversion_rate"] = prod["total_purchases"] / prod["total_views"].replace(0, np.nan)
    prod = prod.fillna(0)

    log.info("Product features built: %d products.", len(prod))
    return prod


# ── Session features ───────────────────────────────────────────────────────────
def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    sess = (
        df.groupby("user_session")
        .agg(
            user_id           = ("user_id", "first"),
            session_events    = ("event_type", "count"),
            session_purchases = ("event_type", lambda x: (x == "purchase").sum()),
            session_revenue   = ("price",      "sum"),
            session_start     = ("event_time", "min"),
            session_end       = ("event_time", "max"),
        )
        .reset_index()
    )
    sess["duration_minutes"] = (
        (sess["session_end"] - sess["session_start"]).dt.total_seconds() / 60
    ).clip(lower=0)
    sess["converted"] = (sess["session_purchases"] > 0).astype(int)

    log.info("Session features built: %d sessions.", len(sess))
    return sess
