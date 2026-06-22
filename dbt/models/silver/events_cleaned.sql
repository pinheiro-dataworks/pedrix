-- Silver: events_cleaned
-- ──────────────────────────────────────────────────────────────────────────────
-- Applies business rules on top of Bronze:
--   · Normalises price (cap negative → 0, fill null → 0)
--   · Derives category_root and category_sub from dot-notation category_code
--   · Generates a surrogate event_id for deduplication
--   · Filters out bot-like sessions (> 500 events per session)
-- ──────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized        = 'incremental',
    schema              = 'silver',
    unique_key          = 'event_id',
    incremental_strategy= 'merge',
    tags                = ['silver', 'cleaned'],
) }}

WITH source AS (
    SELECT * FROM {{ ref('raw_events') }}
    {% if is_incremental() %}
        WHERE event_date > (SELECT MAX(event_date) FROM {{ this }})
    {% endif %}
),

session_sizes AS (
    SELECT
        user_session,
        COUNT(*) AS session_event_count
    FROM source
    GROUP BY user_session
),

cleaned AS (
    SELECT
        -- Surrogate key
        MD5(CONCAT_WS('|',
            COALESCE(s.user_id::VARCHAR,  ''),
            COALESCE(s.user_session,      ''),
            COALESCE(s.event_time::VARCHAR,''),
            COALESCE(s.event_type,        ''),
            COALESCE(s.product_id::VARCHAR,'')
        )) AS event_id,

        s.event_time,
        s.event_type,
        s.product_id,
        s.category_id,
        s.category_code,
        SPLIT_PART(s.category_code, '.', 1) AS category_root,
        SPLIT_PART(s.category_code, '.', 2) AS category_sub,
        COALESCE(s.brand, 'unknown')         AS brand,

        -- Price normalisation
        CASE
            WHEN s.price_raw IS NULL OR s.price_raw < 0 THEN 0.0
            ELSE s.price_raw
        END AS price,

        s.user_id,
        s.user_session,
        s.event_date,
        s.event_month,
        HOUR(s.event_time)         AS event_hour,
        DAYNAME(s.event_time)      AS event_weekday,
        s._ingested_at

    FROM source s
    JOIN session_sizes ss ON s.user_session = ss.user_session
    WHERE ss.session_event_count <= 500  -- bot filter
      AND s.price_raw IS NOT NULL OR s.event_type != 'purchase'
)

SELECT * FROM cleaned
