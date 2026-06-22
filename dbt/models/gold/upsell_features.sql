-- Gold: upsell_features
-- ──────────────────────────────────────────────────────────────────────────────
-- Category-level opportunity scores and user×category affinity weights.
-- Fed into the upsell scoring model (pipeline/models/upsell_model.py).
-- ──────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized = 'table',
    schema       = 'gold',
    tags         = ['gold', 'features', 'ml', 'upsell'],
) }}

WITH events AS (
    SELECT * FROM {{ ref('events_cleaned') }}
),

category_stats AS (
    SELECT
        category_root AS category,
        SUM(CASE WHEN event_type = 'view'     THEN 1 ELSE 0 END) AS total_views,
        SUM(CASE WHEN event_type = 'cart'     THEN 1 ELSE 0 END) AS total_carts,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_revenue,
        AVG(CASE WHEN event_type = 'purchase' THEN price ELSE NULL END) AS avg_price
    FROM events
    WHERE category_root IS NOT NULL AND category_root != 'unknown'
    GROUP BY category_root
),

opportunity AS (
    SELECT
        category,
        total_views,
        total_carts,
        total_purchases,
        total_revenue,
        avg_price,
        CASE WHEN total_views > 0
             THEN total_purchases / total_views::FLOAT
             ELSE 0
        END AS conversion_rate,
        GREATEST(total_views - total_purchases * 8.0, 0) AS opportunity_score,
        CURRENT_TIMESTAMP() AS computed_at
    FROM category_stats
    WHERE total_views >= 100
),

user_category_bought AS (
    SELECT DISTINCT
        user_id,
        category_root AS category
    FROM events
    WHERE event_type = 'purchase'
      AND category_root IS NOT NULL
),

user_category_affinity AS (
    SELECT
        a.user_id,
        b.category AS target_category,
        COUNT(*)   AS shared_buyers
    FROM user_category_bought a
    JOIN user_category_bought b ON a.user_id = b.user_id AND a.category != b.category
    GROUP BY a.user_id, b.category
)

SELECT
    o.category,
    o.total_views,
    o.total_carts,
    o.total_purchases,
    o.total_revenue,
    o.avg_price,
    o.conversion_rate,
    o.opportunity_score,
    o.computed_at
FROM opportunity o
ORDER BY o.opportunity_score DESC
