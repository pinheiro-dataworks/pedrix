-- Gold: churn_features
-- ──────────────────────────────────────────────────────────────────────────────
-- ML-ready feature table for the churn prediction model.
-- Joins RFM scores with view/cart counts and applies the churn label
-- (buyers of month M who did NOT purchase in month M+1).
-- ──────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized = 'table',
    schema       = 'gold',
    tags         = ['gold', 'features', 'ml', 'churn'],
) }}

WITH oct_buyers AS (
    SELECT DISTINCT user_id
    FROM {{ ref('events_cleaned') }}
    WHERE event_type = 'purchase' AND event_month = '2019-10'
),

nov_buyers AS (
    SELECT DISTINCT user_id
    FROM {{ ref('events_cleaned') }}
    WHERE event_type = 'purchase' AND event_month = '2019-11'
),

oct_views AS (
    SELECT user_id, COUNT(*) AS n_views
    FROM {{ ref('events_cleaned') }}
    WHERE event_type = 'view' AND event_month = '2019-10'
    GROUP BY user_id
),

oct_carts AS (
    SELECT user_id, COUNT(*) AS n_carts
    FROM {{ ref('events_cleaned') }}
    WHERE event_type = 'cart' AND event_month = '2019-10'
    GROUP BY user_id
),

rfm AS (
    SELECT * FROM {{ ref('customer_rfm') }}
)

SELECT
    ob.user_id,
    r.recency,
    r.frequency,
    r.monetary,
    r.avg_price,
    r.unique_categories  AS unique_cats,
    r.unique_brands,
    r.unique_products    AS unique_prods,
    COALESCE(v.n_views, 0)                                    AS n_views,
    COALESCE(c.n_carts, 0)                                    AS n_carts,
    CASE
        WHEN r.frequency > 0
        THEN COALESCE(c.n_carts, 0) / r.frequency
        ELSE 0
    END                                                        AS cart_to_pur_ratio,
    r.r_score,
    r.f_score,
    r.m_score,
    r.rfm_score,
    r.segment,
    -- Churn label: 1 if NOT purchased in November
    CASE WHEN nb.user_id IS NULL THEN 1 ELSE 0 END            AS churned,
    CURRENT_TIMESTAMP()                                        AS computed_at

FROM oct_buyers ob
LEFT JOIN rfm         r  ON ob.user_id = r.user_id
LEFT JOIN oct_views   v  ON ob.user_id = v.user_id
LEFT JOIN oct_carts   c  ON ob.user_id = c.user_id
LEFT JOIN nov_buyers  nb ON ob.user_id = nb.user_id
