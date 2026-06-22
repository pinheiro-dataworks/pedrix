-- Bronze: raw_events
-- ──────────────────────────────────────────────────────────────────────────────
-- Thin layer over the Snowflake external stage. Applies schema enforcement only.
-- No business logic here — this table is the single source of truth for raw data.
-- ──────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized = 'table',
    schema       = 'bronze',
    tags         = ['bronze', 'raw'],
) }}

SELECT
    CONVERT_TIMEZONE('UTC', TO_TIMESTAMP_NTZ(event_time))  AS event_time,
    LOWER(TRIM(event_type))                                 AS event_type,
    TRY_CAST(product_id    AS BIGINT)                       AS product_id,
    TRY_CAST(category_id   AS BIGINT)                       AS category_id,
    NULLIF(TRIM(category_code), '')                         AS category_code,
    NULLIF(LOWER(TRIM(brand)), '')                          AS brand,
    TRY_CAST(price AS FLOAT)                                AS price_raw,
    TRY_CAST(user_id AS BIGINT)                             AS user_id,
    NULLIF(TRIM(user_session), '')                          AS user_session,
    -- Partition helpers
    DATE(CONVERT_TIMEZONE('UTC', TO_TIMESTAMP_NTZ(event_time))) AS event_date,
    TO_CHAR(CONVERT_TIMEZONE('UTC', TO_TIMESTAMP_NTZ(event_time)), 'YYYY-MM') AS event_month,
    -- Ingestion metadata
    CURRENT_TIMESTAMP()   AS _ingested_at,
    '{{ var("run_date") }}' AS _run_date

FROM {{ source('landing', 'ecommerce_events') }}

WHERE
    event_time IS NOT NULL
    AND user_id  IS NOT NULL
    AND event_type IN ('view', 'cart', 'purchase')
