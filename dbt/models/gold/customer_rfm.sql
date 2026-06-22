-- Gold: customer_rfm
-- ──────────────────────────────────────────────────────────────────────────────
-- Computes RFM (Recency, Frequency, Monetary) scores for all buyers.
-- Assigns segment labels used by the churn and upsell models.
-- Reference date: {{ var("ref_date") }}
-- ──────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized = 'table',
    schema       = 'gold',
    tags         = ['gold', 'rfm', 'features'],
) }}

WITH purchases AS (
    SELECT *
    FROM {{ ref('events_cleaned') }}
    WHERE event_type = 'purchase'
      AND price > {{ var("min_purchase_price") }}
),

rfm_raw AS (
    SELECT
        user_id,
        DATEDIFF('day', MAX(event_time), TO_TIMESTAMP('{{ var("ref_date") }}')) AS recency,
        COUNT(*)                                                                  AS frequency,
        SUM(price)                                                                AS monetary,
        AVG(price)                                                                AS avg_price,
        COUNT(DISTINCT category_root)                                             AS unique_categories,
        COUNT(DISTINCT brand)                                                     AS unique_brands,
        COUNT(DISTINCT product_id)                                                AS unique_products,
        MIN(event_date)                                                           AS first_purchase_date,
        MAX(event_date)                                                           AS last_purchase_date
    FROM purchases
    GROUP BY user_id
),

rfm_scored AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY recency  ASC)  AS r_score,  -- lower recency = better
        NTILE(5) OVER (ORDER BY frequency DESC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary  DESC) AS m_score
    FROM rfm_raw
    WHERE monetary > 0
),

segmented AS (
    SELECT
        *,
        r_score + f_score + m_score AS rfm_score,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
            WHEN r_score >= 3 AND f_score <= 2 THEN 'Potential Loyalist'
            WHEN r_score  = 2 AND f_score >= 3 THEN 'At Risk'
            WHEN r_score  = 2 AND f_score  = 2 THEN 'Needs Attention'
            WHEN r_score  = 1 AND f_score >= 3 THEN "Can't Lose"
            WHEN r_score  = 1 AND f_score  = 2 THEN 'Hibernating'
            ELSE 'Lost'
        END AS segment
    FROM rfm_scored
)

SELECT
    user_id,
    recency,
    frequency,
    monetary,
    avg_price,
    unique_categories,
    unique_brands,
    unique_products,
    first_purchase_date,
    last_purchase_date,
    r_score,
    f_score,
    m_score,
    rfm_score,
    segment,
    CURRENT_TIMESTAMP() AS computed_at
FROM segmented
