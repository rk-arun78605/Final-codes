-- ============================================================
-- COMPLETE RECOMMENDATIONS MATERIALIZED VIEW
-- ============================================================
-- This view contains ALL business logic:
-- - Priority shop destinations (11 shops)
-- - Source shop selection (non-priority shops with stock > sales)
-- - Sales calculations (MAX of WH GRN +30d vs regular 30d)
-- - Capping with window functions (cumulative never exceeds cap)
-- - GRN age calculations
-- - All blocking rules applied
-- - Final metrics computed
--
-- Python just needs to:
-- 1. SELECT * FROM mv_recommendations_complete WHERE [filters]
-- 2. Format output columns
-- 3. Display in Streamlit
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mv_recommendations_complete CASCADE;

CREATE MATERIALIZED VIEW mv_recommendations_complete AS
WITH 
-- Priority destination shops (ranked 1-11)
priority_shops AS (
    SELECT shop_code, priority_rank
    FROM (VALUES 
        ('SPN', 1), ('MSS', 2), ('LFS', 3), ('M03', 4), ('KAS', 5),
        ('MM1', 6), ('MM2', 7), ('FAR', 8), ('KS7', 9), ('WHL', 10), ('MM3', 11)
    ) AS t(shop_code, priority_rank)
),

-- Item WH GRN dates (item-level, warehouse received date)
item_wh_grn AS (
    SELECT 
        TRIM(UPPER(item_code)) AS item_code,
        MAX(wh_grn_date) AS wh_grn_date,
        MAX(wh_grn_date) + INTERVAL '30 days' AS wh_grn_plus_30
    FROM sup_shop_grn
    WHERE wh_grn_date IS NOT NULL
    GROUP BY 1
),

-- Source shops: Non-priority shops with excess stock (stock > 30d sales)
-- RULE: Priority shops CANNOT be sources
sources AS (
    SELECT 
        im.itemcode AS item_code,
        im.itemname AS item_name,
        im.shopcode AS shop_code,
        im.shopstock AS stock,
        COALESCE(im.sales_30d_wh, 0) AS sales_30d,
        im.shopgrn_dt AS last_grn_date,
        CASE 
            WHEN im.shopgrn_dt IS NOT NULL 
            THEN (CURRENT_DATE - im.shopgrn_dt::date)::integer
            ELSE 0
        END AS grn_age,
        wh.wh_grn_date AS wh_grn_date,
        im.groupp AS groups,
        im.subgroup AS sub_group,
        se."SHOP_EXPIRY_DATE" AS source_expiry_date,
        CASE 
            WHEN se."SHOP_EXPIRY_DATE" IS NOT NULL 
            THEN (se."SHOP_EXPIRY_DATE"::date - CURRENT_DATE )::integer
            ELSE NULL
        END AS source_expiry_days,
        -- Expiry check: Flag if expiry year is more than 2 years from current year
        CASE 
            WHEN se."SHOP_EXPIRY_DATE" IS NULL THEN NULL
            WHEN EXTRACT(YEAR FROM se."SHOP_EXPIRY_DATE"::date) > EXTRACT(YEAR FROM CURRENT_DATE) + 2 
            THEN 'Check expiry date'
            ELSE NULL
        END AS expiry_check,
        im.importexport AS item_type,
        im.suppliername AS supplier_name
    FROM inventory_master im
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(im.itemcode)) = wh.item_code
    LEFT JOIN shopexpiry se ON TRIM(UPPER(im.itemcode)) = TRIM(UPPER(se."ITEM_CODE")) 
        AND TRIM(UPPER(im.shopcode)) = TRIM(UPPER(se."SHOP_CODE"))
    LEFT JOIN itemdetails id ON TRIM(UPPER(im.itemcode)) = TRIM(UPPER(id.vc_item_code))
    WHERE im.shopcode NOT IN (SELECT shop_code FROM priority_shops)  -- NOT priority shops
      AND im.shopstock > COALESCE(im.sales_30d_wh, 0)  -- Excess stock
      AND im.shopstock > 0
),

-- Destinations: Priority shops only
destinations AS (
    SELECT 
        im.itemcode AS item_code,
        im.shopcode AS shop_code,
        im.shopstock AS stock,
        COALESCE(im.sales_30d_wh, 0) AS sales_30d,
        im.shopgrn_dt AS last_grn_date,
        CASE 
            WHEN im.shopgrn_dt IS NOT NULL 
            THEN (CURRENT_DATE - im.shopgrn_dt::date)::integer
            ELSE 0
        END AS grn_age,
        wh.wh_grn_date AS wh_grn_date,
        wh.wh_grn_plus_30 AS wh_grn_plus_30,
        ps.priority_rank
    FROM inventory_master im
    INNER JOIN priority_shops ps ON im.shopcode = ps.shop_code
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(im.itemcode)) = wh.item_code
    WHERE im.itemcode IS NOT NULL
),

-- WH GRN +30d sales for destinations (shop-specific sales in 30 days after WH GRN)
dest_wh_grn_sales AS (
    SELECT 
        d.item_code,
        d.shop_code,
        COALESCE(SUM(s."QTY"), 0) AS wh_grn_30d_sales
    FROM destinations d
    LEFT JOIN sales_2024 s ON 
        TRIM(UPPER(s."ITEM_CODE")) = d.item_code
        AND TRIM(UPPER(s."SHOP_CODE")) = d.shop_code
        AND s."DATE_INVOICE"::date >= d.wh_grn_date::date
        AND s."DATE_INVOICE"::date <= d.wh_grn_plus_30::date
    WHERE d.wh_grn_date IS NOT NULL
    GROUP BY d.item_code, d.shop_code
    
    UNION ALL
    
    SELECT 
        d.item_code,
        d.shop_code,
        COALESCE(SUM(s."QTY"), 0) AS wh_grn_30d_sales
    FROM destinations d
    LEFT JOIN sales_2025 s ON 
        TRIM(UPPER(s."ITEM_CODE")) = d.item_code
        AND TRIM(UPPER(s."SHOP_CODE")) = d.shop_code
        AND s."DATE_INVOICE"::date >= d.wh_grn_date::date
        AND s."DATE_INVOICE"::date <= d.wh_grn_plus_30::date
    WHERE d.wh_grn_date IS NOT NULL
    GROUP BY d.item_code, d.shop_code
),

-- Aggregate WH GRN sales (handle year overlaps)
dest_wh_grn_sales_agg AS (
    SELECT 
        item_code,
        shop_code,
        SUM(wh_grn_30d_sales) AS wh_grn_30d_sales
    FROM dest_wh_grn_sales
    GROUP BY item_code, shop_code
),

-- Destination capacity: MAX(WH GRN +30d sales, regular 30d sales)
dest_capacity AS (
    SELECT 
        d.item_code,
        d.shop_code,
        d.stock,
        d.sales_30d,
        d.last_grn_date,
        d.grn_age,
        d.wh_grn_date,
        d.wh_grn_plus_30,
        d.priority_rank,
        COALESCE(wh.wh_grn_30d_sales, 0) AS wh_grn_30d_sales,
        GREATEST(
            COALESCE(wh.wh_grn_30d_sales, 0),
            d.sales_30d
        ) AS cap
    FROM destinations d
    LEFT JOIN dest_wh_grn_sales_agg wh 
        ON d.item_code = wh.item_code 
        AND d.shop_code = wh.shop_code
),

-- All source-destination combinations (cross join by item)
-- RULE: Source != Destination (same shop blocked)
all_combinations AS (
    SELECT 
        s.item_code,
        s.item_name,
        s.shop_code AS source_shop,
        s.stock AS source_stock,
        s.sales_30d AS source_sales,
        s.last_grn_date AS source_last_grn,
        s.grn_age AS source_grn_age,
        s.wh_grn_date AS source_wh_grn_date,
        s.source_expiry_date,
        s.source_expiry_days,
        s.expiry_check,
        s.groups,
        s.sub_group,
        s.item_type,
        s.supplier_name,
        d.shop_code AS dest_shop,
        d.stock AS dest_stock,
        d.sales_30d AS dest_sales,
        d.last_grn_date AS dest_last_grn,
        d.grn_age AS dest_grn_age,
        d.wh_grn_date AS dest_wh_grn_date,
        d.wh_grn_plus_30 AS dest_wh_grn_plus_30,
        d.wh_grn_30d_sales AS dest_wh_grn_30d_sales,
        d.cap AS dest_cap,
        d.priority_rank
    FROM sources s
    INNER JOIN dest_capacity d ON s.item_code = d.item_code
    WHERE s.shop_code != d.shop_code  -- RULE: Block same shop transfers
),

-- Calculate uncapped recommendations (gap-based allocation)
uncapped_recs AS (
    SELECT 
        *,
        -- Gap: dest_cap - dest_stock (how much destination needs)
        GREATEST(dest_cap - dest_stock, 0) AS dest_gap,
        -- Available: source_stock - source_sales (excess at source)
        GREATEST(source_stock - source_sales, 0) AS source_available,
        -- Uncapped qty: MIN(dest_gap, source_available)
        LEAST(
            GREATEST(dest_cap - dest_stock, 0),
            GREATEST(source_stock - source_sales, 0)
        ) AS uncapped_qty
    FROM all_combinations
    WHERE 
        -- RULE: Only recommend if source has excess
        source_stock > source_sales
        -- RULE: Skip if all three sales metrics are zero (not selling)
        AND NOT (source_sales = 0 AND dest_sales = 0 AND dest_wh_grn_30d_sales = 0)
        -- RULE: Skip destination if stock > 30 but still insufficient (less than both sales metrics)
        -- This filters out low-priority destinations that have some stock but can't meet demand
        AND NOT (
            dest_stock > 30 
            AND dest_stock < LEAST(dest_sales, dest_wh_grn_30d_sales)
        )
),

-- Apply cumulative cap using window function (CRITICAL: ensures total ≤ cap)
-- Orders by: priority_rank (higher priority first), source_grn_age DESC (older stock first)
-- EXPIRY LOGIC: Expired items (source_expiry_days > 0) get recommended_qty = 0, but remain in results
capped_recs AS (
    SELECT 
        item_code,
        item_name,
        source_shop,
        source_stock,
        source_sales,
        source_last_grn,
        source_grn_age,
        source_wh_grn_date,
        source_expiry_date,
        source_expiry_days,
        expiry_check,
        groups,
        sub_group,
        item_type,
        supplier_name,
        dest_shop,
        dest_stock,
        dest_sales,
        dest_last_grn,
        dest_grn_age,
        dest_wh_grn_date,
        dest_wh_grn_plus_30,
        dest_wh_grn_30d_sales,
        dest_cap,
        priority_rank,
        uncapped_qty,
        source_available,
        -- Expiry status classification
        -- Expired: source_expiry_days < 0 (already expired)
        -- Expiring: source_expiry_days between 1 and 30 (will expire soon)
        -- OK: source_expiry_days >= 30 or NULL (safe to transfer)
        CASE 
            WHEN source_expiry_date IS NULL THEN 'OK'
            WHEN source_expiry_days < 0 THEN 'Expired'
            WHEN source_expiry_days BETWEEN 1 AND 30 THEN 'Expiring'
            WHEN source_expiry_days >= 30 THEN 'OK'
            ELSE 'OK'
        END AS expiry_status,
        -- Window function: cumulative sum of uncapped_qty per item+destination
        -- FEFO Ordering: priority rank (ASC), GRN age (DESC - older first), expiry days (ASC - expired/expiring first)
        -- CRITICAL: Only allocate to non-expired items (expiry_status != 'Expired')
        SUM(CASE 
            WHEN source_expiry_date IS NULL THEN uncapped_qty  -- OK items
            WHEN source_expiry_days < 0 THEN 0  -- Expired items excluded from cumulative
            WHEN source_expiry_days BETWEEN 1 AND 30 THEN uncapped_qty  -- Expiring items included
            WHEN source_expiry_days >= 30 THEN uncapped_qty  -- OK items included
            ELSE uncapped_qty
        END) OVER (
            PARTITION BY item_code, dest_shop 
            ORDER BY priority_rank ASC, source_grn_age DESC, 
                     COALESCE(source_expiry_days, 999999) ASC, source_shop ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cumulative_before_this_row,
        -- Capped qty: take what fits within remaining capacity
        -- Formula: MIN(uncapped_qty, MAX(remaining_cap, 0))
        -- CRITICAL: Set to 0 if item is expired (source_expiry_days < 0)
        CASE 
            WHEN source_expiry_date IS NOT NULL AND source_expiry_days < 0 THEN 0  -- Block expired items
            ELSE LEAST(
                uncapped_qty,
                GREATEST(
                    dest_cap - COALESCE(
                        SUM(CASE 
                            WHEN source_expiry_date IS NULL THEN uncapped_qty  -- OK items
                            WHEN source_expiry_days < 0 THEN 0  -- Expired items excluded
                            WHEN source_expiry_days BETWEEN 1 AND 30 THEN uncapped_qty  -- Expiring items included
                            WHEN source_expiry_days >= 30 THEN uncapped_qty  -- OK items included
                            ELSE uncapped_qty
                        END) OVER (
                            PARTITION BY item_code, dest_shop 
                            ORDER BY priority_rank ASC, source_grn_age DESC, 
                                     COALESCE(source_expiry_days, 999999) ASC, source_shop ASC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ),
                        0
                    ),
                    0
                )
            )
        END AS recommended_qty
    FROM uncapped_recs
),

-- Track cumulative allocation per source shop to prevent over-allocation
source_allocation_check AS (
    SELECT 
        *,
        -- Calculate cumulative allocation from this source across ALL destinations
        -- FEFO ordering applied (ASC for expiry_days = expired/expiring first)
        SUM(recommended_qty) OVER (
            PARTITION BY item_code, source_shop
            ORDER BY priority_rank ASC, source_grn_age DESC, 
                     COALESCE(source_expiry_days, 999999) ASC, dest_shop ASC
        ) AS cumulative_source_allocated
    FROM capped_recs
    -- DO NOT FILTER: Keep expired items in results with recommended_qty = 0
),

-- Final output with all metrics - remove duplicates based on all columns except remark
final_results AS (
    SELECT 
        item_code,
        item_name,
        groups,
        sub_group,
        item_type,
        supplier_name,
        source_shop,
        source_stock,
        source_sales,
        source_last_grn,
        source_grn_age,
        source_wh_grn_date,
        source_expiry_date,
        source_expiry_days,
        expiry_check,
        expiry_status,
        dest_shop,
        dest_stock,
        dest_sales,
        dest_last_grn,
        dest_grn_age,
        dest_wh_grn_date,
        dest_wh_grn_plus_30,
        dest_wh_grn_30d_sales,
        dest_cap AS dest_sales_used,
        priority_rank,
        recommended_qty,
        -- Cumulative quantity allocated to this destination
        -- FEFO ordering: priority, GRN age DESC, expiry days ASC (negative/low = expired/expiring first)
        SUM(recommended_qty) OVER (
            PARTITION BY item_code, dest_shop 
            ORDER BY priority_rank ASC, source_grn_age DESC, 
                     COALESCE(source_expiry_days, 999999) ASC, source_shop ASC
        ) AS cumulative_qty,
        -- Remaining cap before this allocation
        -- FEFO ordering applied (ASC for expired/expiring first)
        dest_cap - COALESCE(
            SUM(recommended_qty) OVER (
                PARTITION BY item_code, dest_shop 
                ORDER BY priority_rank ASC, source_grn_age DESC, 
                         COALESCE(source_expiry_days, 999999) ASC, source_shop ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS dest_remaining_cap_before,
        -- Updated stock after transfer
        dest_stock + recommended_qty AS dest_updated_stock,
        -- Final stock days
        -- Formula: (GREATEST(dest_wh_grn_30d_sales, dest_sales) / dest_updated_stock) * 30
        -- Uses higher of WH GRN +30d sales or last 30d sales
        CASE 
            WHEN dest_stock + recommended_qty > 0 AND GREATEST(dest_wh_grn_30d_sales, dest_sales) > 0
            THEN CEIL((dest_stock + recommended_qty) * 30.0 / GREATEST(dest_wh_grn_30d_sales, dest_sales))
            WHEN dest_stock + recommended_qty > 0 THEN 999
            ELSE 0
        END AS dest_final_stock_days,
        -- Remark with expiry consideration
        CASE 
            WHEN expiry_status = 'Expired' THEN 'Item expired - cannot transfer'
            WHEN cumulative_source_allocated >= source_available THEN 'Source stock fully allocated'
            WHEN recommended_qty = 0 AND uncapped_qty > 0 THEN 'Cap reached'
            WHEN recommended_qty > 0 AND recommended_qty < uncapped_qty THEN 'Partial allocation'
            WHEN recommended_qty > 0 THEN 'Full allocation'
            ELSE 'No transfer needed'
        END AS remark
    FROM source_allocation_check
    -- Keep expired items in results (no WHERE filter for expiry_status)
    WHERE cumulative_source_allocated <= source_available  -- RULE: Stop if source is exhausted
)

-- Remove duplicates based on all columns except remark
SELECT DISTINCT ON (
    item_code, item_name, groups, sub_group, item_type, supplier_name,
    source_shop, source_stock, source_sales, source_last_grn, source_grn_age, source_wh_grn_date,
    source_expiry_date, source_expiry_days, expiry_check, expiry_status,
    dest_shop, dest_stock, dest_sales, dest_last_grn, dest_grn_age,
    dest_wh_grn_date, dest_wh_grn_plus_30, dest_wh_grn_30d_sales,
    dest_sales_used, priority_rank, recommended_qty,
    cumulative_qty, dest_remaining_cap_before, dest_updated_stock, dest_final_stock_days
)
    item_code,
    item_name,
    groups,
    sub_group,
    item_type,
    supplier_name,
    source_shop,
    source_stock,
    source_sales,
    source_last_grn,
    source_grn_age,
    source_wh_grn_date,
    source_expiry_date,
    source_expiry_days,
    expiry_check,
    expiry_status,
    dest_shop,
    dest_stock,
    dest_sales,
    dest_last_grn,
    dest_grn_age,
    dest_wh_grn_date,
    dest_wh_grn_plus_30,
    dest_wh_grn_30d_sales,
    dest_sales_used,
    priority_rank,
    recommended_qty,
    cumulative_qty,
    dest_remaining_cap_before,
    dest_updated_stock,
    dest_final_stock_days,
    remark
FROM final_results
ORDER BY 
    item_code, item_name, groups, sub_group, item_type, supplier_name,
    source_shop, source_stock, source_sales, source_last_grn, source_grn_age, source_wh_grn_date,
    source_expiry_date, source_expiry_days, expiry_check, expiry_status,
    dest_shop, dest_stock, dest_sales, dest_last_grn, dest_grn_age,
    dest_wh_grn_date, dest_wh_grn_plus_30, dest_wh_grn_30d_sales,
    dest_sales_used, priority_rank, recommended_qty,
    cumulative_qty, dest_remaining_cap_before, dest_updated_stock, dest_final_stock_days;

-- Create indexes for fast filtering
CREATE INDEX idx_mv_recs_complete_item ON mv_recommendations_complete(item_code);
CREATE INDEX idx_mv_recs_complete_source ON mv_recommendations_complete(source_shop);
CREATE INDEX idx_mv_recs_complete_dest ON mv_recommendations_complete(dest_shop);
CREATE INDEX idx_mv_recs_complete_groups ON mv_recommendations_complete(groups);
CREATE INDEX idx_mv_recs_complete_subgroup ON mv_recommendations_complete(sub_group);
CREATE INDEX idx_mv_recs_complete_item_dest ON mv_recommendations_complete(item_code, dest_shop);

-- Refresh command (run after any data changes)
-- REFRESH MATERIALIZED VIEW mv_recommendations_complete;

COMMENT ON MATERIALIZED VIEW mv_recommendations_complete IS 
'Complete stock transfer recommendations with all business logic:
- Priority shops (11) as destinations only
- Non-priority shops with excess stock as sources
- Cap = MAX(WH GRN +30d sales, regular 30d sales)
- Window function ensures cumulative total ≤ cap per item+destination
- Older stock (higher GRN age) allocated first
- All blocking rules applied (same shop, not selling, etc.)
Python just needs to SELECT with filters and display.';