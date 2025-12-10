-- ============================================================
-- COMPLETE RECOMMENDATIONS MATERIALIZED VIEW - UPDATED Dec 10, 2025
-- ============================================================
-- Business Logic:
-- 1. Each source keeps minimum 30 units as safety stock
-- 2. Only excess above 30 units is transferable
-- 3. Destination cap = MAX(dest_sales_30d, dest_wh_grn_30d_sales)
-- 4. High performers (>50% sales velocity) get priority
-- 5. FEFO ordering (First Expiry First Out - oldest stock first)
-- 6. Block items expiring within 30 days
-- 7. Cumulative allocation per destination ≤ cap
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

-- Get latest expiry date per item+shop
shop_expiry_latest AS (
    SELECT 
        TRIM(UPPER("ITEM_CODE")) AS item_code,
        TRIM(UPPER("SHOP_CODE")) AS shop_code,
        MAX("SHOP_EXPIRY_DATE") AS latest_expiry_date
    FROM shopexpiry
    GROUP BY TRIM(UPPER("ITEM_CODE")), TRIM(UPPER("SHOP_CODE"))
),

-- Source shops: Non-priority shops with stock > 30 units
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
        se.latest_expiry_date AS source_expiry_date,
        CASE 
            WHEN se.latest_expiry_date IS NOT NULL 
            THEN (se.latest_expiry_date::date - CURRENT_DATE)::integer
            ELSE NULL
        END AS source_expiry_days,
        CASE 
            WHEN se.latest_expiry_date IS NULL THEN NULL
            WHEN EXTRACT(YEAR FROM se.latest_expiry_date::date) > EXTRACT(YEAR FROM CURRENT_DATE) + 2 
            THEN 'Check expiry date'
            ELSE NULL
        END AS expiry_check,
        im.importexport AS item_type,
        im.suppliername AS supplier_name
    FROM inventory_master im
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(im.itemcode)) = wh.item_code
    LEFT JOIN shop_expiry_latest se ON TRIM(UPPER(im.itemcode)) = se.item_code 
        AND TRIM(UPPER(im.shopcode)) = se.shop_code
    WHERE im.shopcode NOT IN (SELECT shop_code FROM priority_shops)  -- NOT priority shops
      AND im.shopstock > 30  -- NEW RULE: Must have more than 30 units to be a source
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

-- WH GRN +30d sales for destinations
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
        -- Cap = MAX of the two sales metrics
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
    WHERE s.shop_code != d.shop_code  -- Block same shop transfers
),

-- Calculate uncapped recommendations
-- NEW RULE: Each source keeps minimum 30 units, only excess above 30 is transferable
uncapped_recs AS (
    SELECT 
        *,
        -- Gap: How much destination needs
        GREATEST(dest_cap - dest_stock, 0) AS dest_gap,
        
        -- NEW LOGIC: Source available = stock - 30 (keep 30 units as safety stock)
        GREATEST(source_stock - 30, 0) AS source_available,
        
        -- High performer flag: Destination selling >50% of cap
        CASE 
            WHEN dest_cap > 0 AND (GREATEST(dest_sales, dest_wh_grn_30d_sales) / dest_cap) > 0.5 
            THEN TRUE
            ELSE FALSE
        END AS dest_high_performer,
        
        -- Uncapped qty: MIN(dest_gap, source_available after keeping 30)
        LEAST(
            GREATEST(dest_cap - dest_stock, 0),
            GREATEST(source_stock - 30, 0)
        ) AS uncapped_qty
    FROM all_combinations
    WHERE 
        -- Source must have MORE than 30 units (already filtered in sources CTE)
        source_stock > 30
        -- Skip if no one is selling (not moving)
        AND NOT (source_sales = 0 AND dest_sales = 0 AND dest_wh_grn_30d_sales = 0)
        -- Destination must have positive gap (needs stock)
        AND dest_cap > dest_stock
),

-- Apply cumulative cap with FEFO ordering
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
        dest_high_performer,
        uncapped_qty,
        source_available,
        
        -- Expiry status
        CASE 
            WHEN source_expiry_date IS NULL THEN 'OK'
            WHEN source_expiry_days < 0 THEN 'Expired'
            WHEN source_expiry_days BETWEEN 1 AND 29 THEN 'Expiring'
            WHEN source_expiry_days >= 30 THEN 'OK'
            ELSE 'OK'
        END AS expiry_status,
        
        -- Cumulative allocation before this row (per item+destination)
        -- Ordering: priority_rank ASC, high_performers first, then FEFO (GRN age DESC)
        SUM(CASE 
            WHEN source_expiry_date IS NULL THEN uncapped_qty
            WHEN source_expiry_days < 0 THEN 0
            WHEN source_expiry_days BETWEEN 1 AND 29 THEN 0
            WHEN source_expiry_days >= 30 THEN uncapped_qty
            ELSE 0
        END) OVER (
            PARTITION BY item_code, dest_shop 
            ORDER BY priority_rank ASC, 
                     dest_high_performer DESC,  -- High performers get priority
                     source_grn_age DESC,       -- FEFO: Oldest first
                     COALESCE(source_expiry_days, 999999) ASC, 
                     source_shop ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cumulative_before_this_row,
        
        -- Recommended qty: Capped to not exceed destination cap
        CASE 
            WHEN source_expiry_date IS NOT NULL AND source_expiry_days < 30 THEN 0
            ELSE LEAST(
                uncapped_qty,
                GREATEST(
                    dest_cap - COALESCE(
                        SUM(CASE 
                            WHEN source_expiry_date IS NULL THEN uncapped_qty
                            WHEN source_expiry_days < 0 THEN 0
                            WHEN source_expiry_days BETWEEN 1 AND 29 THEN 0
                            WHEN source_expiry_days >= 30 THEN uncapped_qty
                            ELSE 0
                        END) OVER (
                            PARTITION BY item_code, dest_shop 
                            ORDER BY priority_rank ASC, 
                                     dest_high_performer DESC,
                                     source_grn_age DESC, 
                                     COALESCE(source_expiry_days, 999999) ASC, 
                                     source_shop ASC
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

-- Track cumulative allocation per source shop
source_allocation_check AS (
    SELECT 
        *,
        -- Total allocated FROM this source TO all destinations
        SUM(recommended_qty) OVER (
            PARTITION BY item_code, source_shop
            ORDER BY priority_rank ASC, 
                     dest_high_performer DESC,
                     source_grn_age DESC, 
                     COALESCE(source_expiry_days, 999999) ASC, 
                     dest_shop ASC
        ) AS cumulative_source_allocated
    FROM capped_recs
),

-- Final results with all metrics
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
        SUM(recommended_qty) OVER (
            PARTITION BY item_code, dest_shop 
            ORDER BY priority_rank ASC, 
                     dest_high_performer DESC,
                     source_grn_age DESC, 
                     COALESCE(source_expiry_days, 999999) ASC, 
                     source_shop ASC
        ) AS cumulative_qty,
        
        -- Remaining cap before this allocation
        dest_cap - COALESCE(
            SUM(recommended_qty) OVER (
                PARTITION BY item_code, dest_shop 
                ORDER BY priority_rank ASC, 
                         dest_high_performer DESC,
                         source_grn_age DESC, 
                         COALESCE(source_expiry_days, 999999) ASC, 
                         source_shop ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS dest_remaining_cap_before,
        
        -- Updated stock after transfer
        dest_stock + recommended_qty AS dest_updated_stock,
        
        -- Final stock days
        CASE 
            WHEN dest_stock + recommended_qty > 0 AND GREATEST(dest_wh_grn_30d_sales, dest_sales) > 0
            THEN CEIL((dest_stock + recommended_qty) * 30.0 / GREATEST(dest_wh_grn_30d_sales, dest_sales))
            WHEN dest_stock + recommended_qty > 0 THEN 999
            ELSE 0
        END AS dest_final_stock_days,
        
        -- Remark
        CASE 
            WHEN expiry_status = 'Expired' THEN 'Item expired - cannot transfer'
            WHEN expiry_status = 'Expiring' AND source_expiry_days < 30 THEN 'Expiring soon - cannot transfer'
            WHEN cumulative_source_allocated >= source_available THEN 'Source stock fully allocated (keeping 30 units)'
            WHEN recommended_qty = 0 AND uncapped_qty > 0 THEN 'Cap reached'
            WHEN recommended_qty > 0 AND recommended_qty < uncapped_qty THEN 'Partial allocation'
            WHEN recommended_qty > 0 THEN 'Full allocation'
            ELSE 'No transfer needed'
        END AS remark
    FROM source_allocation_check
)

-- Return final results
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
    item_code,
    dest_shop,
    priority_rank ASC,
    source_grn_age DESC,
    COALESCE(source_expiry_days, 999999) ASC,
    source_shop ASC;

-- Create indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_mv_recs_complete_item ON mv_recommendations_complete(item_code);
CREATE INDEX IF NOT EXISTS idx_mv_recs_complete_source ON mv_recommendations_complete(source_shop);
CREATE INDEX IF NOT EXISTS idx_mv_recs_complete_dest ON mv_recommendations_complete(dest_shop);
CREATE INDEX IF NOT EXISTS idx_mv_recs_complete_groups ON mv_recommendations_complete(groups);
CREATE INDEX IF NOT EXISTS idx_mv_recs_complete_subgroup ON mv_recommendations_complete(sub_group);
CREATE INDEX IF NOT EXISTS idx_mv_recs_complete_item_dest ON mv_recommendations_complete(item_code, dest_shop);

COMMENT ON MATERIALIZED VIEW mv_recommendations_complete IS 
'Stock transfer recommendations - UPDATED Dec 10, 2025:
BUSINESS RULES:
1. Each source keeps minimum 30 units as safety stock
2. Only excess above 30 units available for transfer (source_available = stock - 30)
3. Destination cap = MAX(dest_sales_30d, dest_wh_grn_30d_sales)
4. High performers (>50% sales velocity) get priority allocation
5. FEFO ordering (First Expiry First Out - oldest stock first)
6. Block items expiring within 30 days
7. Cumulative allocation per destination ≤ cap
8. Priority shops (11): SPN, MSS, LFS, M03, KAS, MM1, MM2, FAR, KS7, WHL, MM3';
