# melcom_inventory_pg.py
"""
Melcom NO_WH Inventory Pulse - Production Ready
PostgreSQL-based Inventory Management & Stock Transfer Recommendation System
Version: 2.3 - Fixed dynamic sales calculation with debugging
"""

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import io
import warnings
import plotly.graph_objects as go
import time
import hashlib
from psycopg2 import pool, Error
from psycopg2.extras import RealDictCursor
#from io import BytesIO
from typing import Optional, Dict, Tuple
from contextlib import contextmanager
import logging
from datetime import datetime, timedelta

# Suppress pandas SQLAlchemy warnings
warnings.filterwarnings('ignore', message='.*SQLAlchemy.*')

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Centralized configuration"""
    
    # Database configurations
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'postgres',
        'password': 'hello',
        'port': 3307
    }
    
    # Business constants
    PRIORITY_SHOPS = ['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']
    ALL_SHOPS = [
        'SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3',
        'ACC', 'ACH', 'ADB', 'AF2', 'AFI', 'AFL', 'AKE', 'AMA', 'ASF', 'ASH', 'AWS',
        'BIB', 'BOL', 'BRE', 'CAP', 'CLC', 'DNS', 'ELS', 'GBA', 'HAA', 'HAM', 'HOE',
        'HOV', 'KA2', 'KCC', 'KCS', 'KF2', 'KFD', 'KNS', 'KS2', 'KS3', 'KS5', 'KS8',
        'KSI', 'KSO', 'KSS', 'LCC', 'M02', 'M04', 'M05', 'M06', 'M07', 'MAS', 'MCC',
        'MDN', 'MKC', 'MKL', 'MUS', 'NAN', 'NKW', 'OLE', 'SCC', 'SD2', 'SDR', 'SPX',
        'SU2', 'SUC', 'SUN', 'SWO', 'TC2', 'TCC', 'THC', 'TKD', 'TKW', 'TM2', 'TMA',
        'TML', 'TMP', 'WNC'
    ]
    DEFAULT_THRESHOLD = 30  # Changed from 10 to 1
    BUFFER_DAYS = 30
    GRN_FALLBACK_DAYS = 90
    SALES_DAYS_WINDOW = 30
    
    # UI
    PAGE_TITLE = "Inventory Pulse NO_WH"
    PAGE_ICON = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'rec_cache' not in st.session_state:
    st.session_state.rec_cache = {}  # In-memory cache for recommendations

# ============================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================

@st.cache_resource
def get_connection_pool(dbname: str):
    """Create connection pool for database"""
    try:
        return psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dbname=dbname,
            connect_timeout=5,
            **Config.DB_CONFIG
        )
    except Error as e:
        logger.error(f"Connection pool error for {dbname}: {e}")
        st.error(f"❌ Cannot create connection pool for {dbname}: {e}")
        return None

@contextmanager
def get_db_connection(dbname: str):
    """Context manager for safe database connections"""
    pool = get_connection_pool(dbname)
    if not pool:
        raise Exception(f"Connection pool not available for {dbname}")
    
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    except Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            pool.putconn(conn)

# ============================================================
# AUTHENTICATION
# ============================================================

@st.cache_data(ttl=3600, show_spinner=True)
def authenticate_user(employee_id: str, password: str) -> Optional[Dict]:
    """Authenticate user from PostgreSQL users database (cached for speed)"""
    try:
        with get_db_connection('users') as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT employee_id, full_name, db_access, table_access, is_active
                    FROM users
                    WHERE employee_id = %s AND password = %s AND LOWER(is_active) = 'true'
                """, (employee_id, password))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Failed login: {employee_id}")
                    return None
                
                logger.info(f"✅ Successful login: {employee_id} (cached)")
                return dict(row)
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

def check_table_access(user: Dict, required_table: str) -> bool:
    """Check if user has access to required table"""
    if not user or 'table_access' not in user:
        return False
    
    user_tables = [t.strip().lower() for t in (user.get('table_access') or '').split(',')]
    return required_table.lower() in user_tables or 'all' in user_tables

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_data_freshness() -> Dict[str, str]:
    """Get latest data update dates from all source tables (cached 5 min)"""
    freshness = {'inventory': 'N/A', 'grn': 'N/A', 'sales': 'N/A'}
    
    try:
        # Get Inventory and GRN dates from grndetails database
        try:
            with get_db_connection('grndetails') as conn:
                # Get inventory date (from inventory_master table)
                query_inv = 'SELECT MAX(shopgrn_dt) as latest_date FROM inventory_master WHERE shopgrn_dt IS NOT NULL'
                result = pd.read_sql(query_inv, conn)
                if not result.empty and pd.notna(result['latest_date'].iloc[0]):
                    date_val = result['latest_date'].iloc[0]
                    if isinstance(date_val, str):
                        freshness['inventory'] = pd.to_datetime(date_val).strftime('%d-%b-%Y')
                    else:
                        freshness['inventory'] = date_val.strftime('%d-%b-%Y')
                
                # Get GRN date (from sup_shop_grn table)
                query_grn = 'SELECT MAX(shop_grn_date) as latest_date FROM sup_shop_grn WHERE shop_grn_date IS NOT NULL'
                result = pd.read_sql(query_grn, conn)
                if not result.empty and pd.notna(result['latest_date'].iloc[0]):
                    date_val = result['latest_date'].iloc[0]
                    if isinstance(date_val, str):
                        freshness['grn'] = pd.to_datetime(date_val).strftime('%d-%b-%Y')
                    else:
                        freshness['grn'] = date_val.strftime('%d-%b-%Y')
        except Exception as e:
            logger.error(f"Error getting inventory/GRN dates: {e}", exc_info=True)
        
        # Sales data (sales_2025 - in salesdata db)
        try:
            with get_db_connection('salesdata') as conn:
                query_sales = 'SELECT MAX("DATE_INVOICE") as latest_date FROM sales_2025 WHERE "DATE_INVOICE" IS NOT NULL'
                result = pd.read_sql(query_sales, conn)
                if not result.empty and pd.notna(result['latest_date'].iloc[0]):
                    date_val = result['latest_date'].iloc[0]
                    if isinstance(date_val, str):
                        freshness['sales'] = pd.to_datetime(date_val).strftime('%d-%b-%Y')
                    else:
                        freshness['sales'] = date_val.strftime('%d-%b-%Y')
        except Exception as e:
            logger.error(f"Error getting sales date: {e}", exc_info=True)
        
        logger.info(f"✅ Data freshness: Inventory={freshness['inventory']}, GRN={freshness['grn']}, Sales={freshness['sales']}")
        return freshness
    except Exception as e:
        logger.error(f"Error in get_data_freshness: {e}", exc_info=True)
        return {'inventory': 'Error', 'grn': 'Error', 'sales': 'Error'}

@st.cache_data(ttl=300, show_spinner=False)
def load_filter_options() -> pd.DataFrame:
    """Load distinct filter options with groups/sub_group from itemdetails (cached 5 min)"""
    try:
        with get_db_connection('salesdata') as conn:
            query = '''
                SELECT DISTINCT 
                    COALESCE(id.groups, im.groupp, 'N/A') AS "GROUPS", 
                    COALESCE(id.sub_group, im.subgroup, 'N/A') AS "SUB_GROUP", 
                    im.itemcode AS "ITEM_CODE", 
                    im.itemname AS "ITEM_NAME", 
                    im.shopcode AS "SHOP_CODE"
                FROM inventory_master im
                LEFT JOIN itemdetails id ON TRIM(UPPER(im.itemcode)) = TRIM(UPPER(id.vc_item_code))
            '''
            df = pd.read_sql(query, conn)
            logger.info(f"✅ Loaded filter options with itemdetails join: {len(df)} records (cached)")
            
            if df.empty:
                st.warning("⚠️ No filter data found in inventory_master table")
                logger.warning("Filter query returned empty dataframe")
            
            return df
    except Exception as e:
        logger.error(f"Error loading filters: {e}")
        st.error(f"❌ Database Error: Could not load filter options. Please check database connection.\n\nDetails: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sit_filter_options() -> pd.DataFrame:
    """Load distinct filter options from inventory_master + itemdetails (cached 1 hour)"""
    try:
        with get_db_connection('salesdata') as conn:
            query = '''
                SELECT DISTINCT 
                    TRIM(UPPER(im.itemcode)) as item_code,
                    im.itemname as item_name,
                    COALESCE(id.type, 'Unknown') as type,
                    im.suppliername as vc_supplier_name
                FROM inventory_master im
                LEFT JOIN itemdetails id ON TRIM(UPPER(id.vc_item_code)) = TRIM(UPPER(im.itemcode))
                WHERE im.itemname IS NOT NULL AND im.itemname != ''
                  AND im.suppliername IS NOT NULL AND im.suppliername != ''
            '''
            df = pd.read_sql(query, conn)
            logger.info(f"✅ Loaded item filter options: {len(df)} items from inventory_master + itemdetails")
            return df
    except Exception as e:
        logger.error(f"Error loading item filters: {e}")
        logger.warning(f"⚠️ Item metadata not available: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sit_filter_lookup() -> Dict:
    """Load and normalize item filter options into fast lookup dicts (cached 1 hour)"""
    try:
        sit_df = load_sit_filter_options()
        if sit_df.empty:
            return {}
        
        import time
        t0 = time.time()
        
        # Normalize once (columns are already uppercase from query)
        sit_df = sit_df.copy()
        sit_df['ITEM_CODE'] = sit_df['item_code'].astype(str).str.strip().str.upper()
        sit_df['ITEM_NAME_SIT'] = sit_df['item_name'].astype(str).str.strip()
        sit_df['TYPE_SIT'] = sit_df['type'].astype(str).str.strip()
        sit_df['SUPPLIER_SIT'] = sit_df['vc_supplier_name'].astype(str).str.strip()
        
        # Build fast lookup: (type, supplier, item_name) -> set of item_codes
        lookup = {}
        for _, row in sit_df.iterrows():
            key = (row['TYPE_SIT'], row['SUPPLIER_SIT'], row['ITEM_NAME_SIT'])
            if key not in lookup:
                lookup[key] = set()
            lookup[key].add(row['ITEM_CODE'])
        
        elapsed = time.time() - t0
        logger.info(f"✅ Built item filter lookup ({len(lookup)} combinations) in {elapsed:.3f}s")
        return lookup
    except Exception as e:
        logger.error(f"Error building item filter lookup: {e}")
        return {}


def apply_itemdetails_filters(inventory_df: pd.DataFrame, sit_df: pd.DataFrame, item_type: str, supplier: str, item_name: str) -> pd.DataFrame:
    """Filter `inventory_df` using SIT item details selections (fast vectorized version).

    Uses pre-computed lookup for O(1) filter matching instead of repeated normalizations.
    If no SIT filters selected, returns inventory_df unchanged.
    """
    try:
        import time
        t0 = time.time()
        
        # If all filters are 'All', skip filtering
        if (not item_type or item_type == 'All') and (not supplier or supplier == 'All') and (not item_name or item_name == 'All'):
            logger.info(f"✅ SIT filters all 'All' - returning full inventory ({len(inventory_df)} rows)")
            return inventory_df
        
        if sit_df is None or sit_df.empty:
            return inventory_df
        
        # Get pre-computed lookup
        sit_lookup = load_sit_filter_lookup()
        if not sit_lookup:
            return inventory_df
        
        # Collect matching item codes using set union
        matched_codes = set()
        for (ftype, fsupplier, fname), codes in sit_lookup.items():
            type_match = (not item_type or item_type == 'All' or ftype == item_type)
            supplier_match = (not supplier or supplier == 'All' or fsupplier == supplier)
            name_match = (not item_name or item_name == 'All' or fname == item_name)
            
            if type_match and supplier_match and name_match:
                matched_codes.update(codes)
        
        if not matched_codes:
            logger.warning(f"⚠️ SIT filters matched 0 items (type={item_type}, supplier={supplier}, item_name={item_name})")
            return inventory_df.iloc[0:0]
        
        # Fast filter using set membership (no normalization needed - keys already normalized)
        inv = inventory_df.copy()
        inv['ITEM_CODE'] = inv['ITEM_CODE'].astype(str).str.strip().str.upper()
        result = inv[inv['ITEM_CODE'].isin(matched_codes)].reset_index(drop=True)
        
        elapsed = time.time() - t0
        logger.info(f"✅ SIT filter applied: {len(inventory_df)} -> {len(result)} rows ({elapsed:.3f}s)")
        return result

    except Exception as e:
        logger.error(f"Error applying item details filters: {e}")
        return inventory_df

@st.cache_data(ttl=900, show_spinner=False)
def load_last_30d_sales() -> pd.DataFrame:
    """Load sales from last 30 days (cached 15 min) - dynamically handles year overlaps"""
    try:
        end_date = datetime.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=Config.SALES_DAYS_WINDOW - 1)
        
        # Determine which years to query
        start_year = start_date.year
        end_year = end_date.year
        years_to_query = list(range(start_year, end_year + 1))
        
        logger.info(f"📅 Loading 30d sales from {start_date.date()} to {end_date.date()} (years: {years_to_query})")
        
        with get_db_connection('salesdata') as conn:
            sales_dfs = []
            
            for year in years_to_query:
                table_name = f"sales_{year}"
                try:
                    query = f"""
                        SELECT 
                            TRIM(UPPER("ITEM_CODE")) AS item_code,
                            TRIM(UPPER("SHOP_CODE")) AS shop_code,
                            SUM(COALESCE("QTY", 0))::INT AS sales_30d
                        FROM {table_name}
                        WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
                        GROUP BY 1, 2
                    """
                    year_sales = pd.read_sql(query, conn, params=(start_date.date(), end_date.date()))
                    sales_dfs.append(year_sales)
                    logger.info(f"✅ Loaded {len(year_sales)} rows from {table_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {table_name}: {e}")
            
            if not sales_dfs:
                return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SALES_30D"])
            
            # Combine all years and aggregate
            sales_df = pd.concat(sales_dfs, ignore_index=True)
            sales_df = sales_df.groupby(['item_code', 'shop_code'], as_index=False)['sales_30d'].sum()
        
        sales_df.columns = sales_df.columns.str.upper()
        logger.info(f"✅ Total 30-day sales: {len(sales_df)} rows (cached)")
        return sales_df if not sales_df.empty else pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SALES_30D"])
        
    except Exception as e:
        logger.error(f"Error loading 30-day sales: {e}")
        return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SALES_30D"])

@st.cache_data(ttl=900, show_spinner=False)
def load_shop_grn_sales(shop_code: str = None) -> pd.DataFrame:
    """
    Load shop's last 30-day sales based on last WH GRN date from sup_shop_grn.
    Handles year overlaps dynamically (e.g., Dec 20 GRN spans into next year).
    
    CRITICAL FIX: WH GRN date is ITEM-LEVEL (warehouse-level), not shop-specific.
    The same item has the same WH GRN date across ALL shops.
    
    Args:
        shop_code: Specific shop to load (None = all shops)
    
    Returns:
        DataFrame with columns: ITEM_CODE, SHOP_CODE, SHOP_GRN_SALES_30D
    """
    try:
        with get_db_connection('salesdata') as conn:
            # CRITICAL: Get WH GRN date PER ITEM ONLY (not per shop)
            # WH GRN is when warehouse received the item - same for all shops
            grn_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER(item_code)) AS item_code,
                    MAX(wh_grn_date) AS wh_last_grn_date
                FROM sup_shop_grn
                WHERE wh_grn_date IS NOT NULL
                GROUP BY 1
            """, conn)
            
            if grn_df.empty:
                logger.warning(f"⚠️ No GRN data found")
                return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SHOP_GRN_SALES_30D"])
            
            # Convert to datetime
            grn_df["wh_last_grn_date"] = pd.to_datetime(grn_df["wh_last_grn_date"], errors="coerce")
            grn_df = grn_df[grn_df["wh_last_grn_date"].notna()]
            
            if grn_df.empty:
                logger.warning(f"⚠️ No valid GRN dates after parsing")
                return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SHOP_GRN_SALES_30D"])
            
            # Calculate 30 days from WH GRN date
            grn_df['grn_30d_end'] = grn_df['wh_last_grn_date'] + pd.Timedelta(days=30)
            
            logger.info(f"✅ Loaded WH GRN dates for {len(grn_df)} unique items")
            
            # Determine years needed
            grn_years = grn_df['wh_last_grn_date'].dt.year.unique()
            end_years = grn_df['grn_30d_end'].dt.year.unique()
            all_years = sorted(set(list(grn_years) + list(end_years)))
            
            logger.info(f"📅 WH GRN dates span years: {all_years}")
            
            # Apply shop filter if specified
            shop_filter_sql = ""
            if shop_code:
                shop_filter_sql = f"WHERE TRIM(UPPER(\"SHOP_CODE\")) = '{shop_code.upper()}'"
            
            # Load sales data from all required years
            sales_dfs = []
            for year in all_years:
                table_name = f"sales_{year}"
                try:
                    year_sales_df = pd.read_sql(f"""
                        SELECT 
                            TRIM(UPPER("ITEM_CODE")) AS item_code,
                            TRIM(UPPER("SHOP_CODE")) AS shop_code,
                            "DATE_INVOICE"::date AS date_invoice,
                            SUM(COALESCE("QTY", 0)) AS qty
                        FROM {table_name}
                        {shop_filter_sql}
                        GROUP BY 1, 2, 3
                    """, conn)
                    
                    sales_dfs.append(year_sales_df)
                    logger.info(f"✅ Loaded {len(year_sales_df)} sales records from {table_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {table_name}: {e}")
            
            if not sales_dfs:
                logger.error("❌ No sales data loaded from any year")
                return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SHOP_GRN_SALES_30D"])
            
            # Combine all years
            sales_df = pd.concat(sales_dfs, ignore_index=True)
            sales_df["date_invoice"] = pd.to_datetime(sales_df["date_invoice"], errors="coerce")
            
            # CRITICAL: Merge on item_code only - WH GRN is item-level
            # This creates a cross join: each item's WH GRN date × all shops that sold it
            merged = sales_df.merge(grn_df, on="item_code", how="inner")
            
            # Filter sales within 30 days of WH GRN date
            filtered = merged[
                (merged["date_invoice"] >= merged["wh_last_grn_date"]) & 
                (merged["date_invoice"] <= merged["grn_30d_end"])
            ]
            
            logger.info(f"📊 Filtered to {len(filtered)} sales records within WH GRN +30 day window")
            
            # Aggregate sales by item-shop
            result = (
                filtered.groupby(["item_code", "shop_code"])["qty"]
                .sum()
                .reset_index()
                .rename(columns={"qty": "shop_grn_sales_30d"})
            )
            
            result.columns = result.columns.str.upper()
            logger.info(f"✅ Calculated WH GRN-based 30d sales for {len(result)} item-shop combinations")
            return result
            
    except Exception as e:
        logger.error(f"❌ Error loading shop GRN sales: {e}")
        return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SHOP_GRN_SALES_30D"])


@st.cache_data(ttl=900, show_spinner=False)
def load_wh_grn_sales() -> pd.DataFrame:
    """
    Load WH last 30-day sales based on last GRN date (kept for backward compatibility).
    Now just calls load_shop_grn_sales() for WHL shop.
    """
    result = load_shop_grn_sales('WHL')
    if not result.empty:
        # Rename for backward compatibility
        result = result.rename(columns={"SHOP_GRN_SALES_30D": "WH_GRN_SALES_30D"})
        result = result.drop(columns=["SHOP_CODE"], errors="ignore")
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def load_inventory_data(group: str, subgroup: str, product: str, shop: str) -> pd.DataFrame:
    """Load inventory data from optimized inventory_master table with filters
    
    OPTIMIZED: Now includes pre-computed WH GRN sales for faster recommendations
    """
    try:
        with get_db_connection('salesdata') as conn:
            # Build filters
            filters = []
            params = []
            if group and group.strip() != "" and group != "All":
                filters.append('groupp = %s')
                params.append(group)
            if subgroup and subgroup.strip() != "" and subgroup != "All":
                filters.append('subgroup = %s')
                params.append(subgroup)
            if product and product.strip() != "" and product != "All":
                filters.append('itemcode = %s')
                params.append(product)
            if shop and shop.strip() != "" and shop != "All":
                filters.append('shopcode = %s')
                params.append(shop)

            where_sql = "WHERE " + " AND ".join(filters) if filters else ""

            # Use optimized inventory_master table with itemdetails join for groups/sub_group
            # CRITICAL: whgrn_dt is the WH GRN date (item-level, warehouse received date)
            query = f"""
                SELECT 
                    im.itemcode AS "ITEM_CODE", 
                    im.itemname AS "ITEM_NAME", 
                    im.shopcode AS "SHOP_CODE",
                    im.shopstock AS "SHOP_STOCK", 
                    COALESCE(id.groups, im.groupp, 'N/A') AS "GROUPS", 
                    COALESCE(id.sub_group, im.subgroup, 'N/A') AS "SUB_GROUP", 
                    im.shopgrn_dt AS "SHOP_GRN_DATE",
                    im.whgrn_dt AS "WH_GRN_DATE",
                    im.dept AS "DEPARTMENT",
                    im.sales_30d_wh AS "ITEM_SALES_30_DAYS"
                FROM inventory_master im
                LEFT JOIN itemdetails id ON TRIM(UPPER(im.itemcode)) = TRIM(UPPER(id.vc_item_code))
                {where_sql}
            """

            df = pd.read_sql(query, conn, params=params)
            
            # Optimized data cleaning
            df['ITEM_CODE'] = df['ITEM_CODE'].str.strip().str.upper()
            df['SHOP_CODE'] = df['SHOP_CODE'].str.strip().str.upper()
            df['SHOP_STOCK'] = pd.to_numeric(df['SHOP_STOCK'], errors='coerce').fillna(0).astype(int)
            df['ITEM_SALES_30_DAYS'] = pd.to_numeric(df.get('ITEM_SALES_30_DAYS', 0), errors='coerce').fillna(0).astype(int)
            
            # inventory_master already has sales_30d_wh, no need to merge separately
            logger.info(f"✅ Loaded {len(df)} records from inventory_master (includes sales data)")
        
        logger.info(f"✅ Loaded {len(df)} inventory records")
        return df
            
    except Exception as e:
        logger.error(f"Error loading inventory: {e}")
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def load_last_grn_dates() -> pd.DataFrame:
    """Load last GRN date per item-shop from materialized view (cached 15 min)"""
    try:
        with get_db_connection('grndetails') as conn:
            query = """
                SELECT 
                    item_code,
                    shop_code,
                    last_grn_date
                FROM mv_last_grn_dates
            """
            df = pd.read_sql(query, conn)
            if df.empty:
                logger.warning("⚠️ mv_last_grn_dates returned no rows")
                return pd.DataFrame(columns=['ITEM_CODE', 'SHOP_CODE', 'LAST_GRN_DATE'])
            
            # Normalize columns to uppercase for consistency
            df.columns = df.columns.str.upper()
            logger.info(f"✅ Loaded last GRN dates: {len(df)} item-shop combinations (cached)")
            return df
    except Exception as e:
        logger.error(f"❌ Error loading last GRN dates: {e}")
        logger.warning(f"⚠️ Using fallback: will calculate GRN dates from sup_shop_grn table instead")
        # Fallback: query the base table directly
        try:
            with get_db_connection('grndetails') as conn:
                query = """
                    SELECT 
                        TRIM(UPPER("ITEM_CODE")) AS ITEM_CODE,
                        TRIM(UPPER("SHOP_CODE")) AS SHOP_CODE,
                        MAX("SHOP_GRN_DATE") AS LAST_GRN_DATE
                    FROM sup_shop_grn
                    GROUP BY 1, 2
                """
                df = pd.read_sql(query, conn)
                logger.info(f"✅ Fallback: Loaded {len(df)} GRN dates from sup_shop_grn")
                return df
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {e2}")
            return pd.DataFrame(columns=['ITEM_CODE', 'SHOP_CODE', 'LAST_GRN_DATE'])

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_grn_sales() -> pd.DataFrame:
    """Calculate sales since last GRN date"""
    try:
        with get_db_connection('grndetails') as conn_grn, get_db_connection('salesdata') as conn_sales:
            # Load GRN dates
            grn_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER("ITEM_CODE")) AS item_code,
                    TRIM(UPPER("SHOP_CODE")) AS shop_code,
                    MAX("SHOP_GRN_DATE")::date AS last_grn_date
                FROM sup_shop_grn
                GROUP BY 1, 2
            """, conn_grn)
            
            # Load sales
            sales_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER("ITEM_CODE")) AS item_code,
                    TRIM(UPPER("SHOP_CODE")) AS shop_code,
                    "DATE_INVOICE"::date AS date_invoice,
                    SUM(COALESCE("QTY", 0)) AS qty
                FROM sales_2025
                GROUP BY 1, 2, 3
            """, conn_sales)
        
        # Process dates
        grn_df["last_grn_date"] = pd.to_datetime(grn_df["last_grn_date"], errors="coerce")
        sales_df["date_invoice"] = pd.to_datetime(sales_df["date_invoice"], errors="coerce")
        
        # Merge and calculate
        merged = sales_df.merge(grn_df, on=["item_code", "shop_code"], how="left")
        fallback = pd.Timestamp.today() - pd.Timedelta(days=Config.GRN_FALLBACK_DAYS)
        merged["last_grn_date"] = merged["last_grn_date"].fillna(fallback)
        
        # Filter sales since GRN
        filtered = merged[merged["date_invoice"] >= merged["last_grn_date"]]
        
        # Aggregate
        result = (
            filtered.groupby(["item_code", "shop_code"])["qty"]
            .sum()
            .reset_index()
            .rename(columns={"qty": "total_sales_grn_to_today"})
        )
        
        # Normalize column names
        result.columns = result.columns.str.upper()
        logger.info(f"Calculated GRN sales for {len(result)} combinations")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating GRN sales: {e}")
        st.warning(f"⚠️ GRN sales calculation failed: {e}")
        return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "TOTAL_SALES_GRN_TO_TODAY"])

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================


def format_recommendations(df: pd.DataFrame, skip_capping: bool = False) -> pd.DataFrame:
    """
    Format raw recommendation query results into final output structure.
    SIMPLIFIED: When skip_capping=True, just format the output - NO business logic!
    """
    if df.empty:
        return pd.DataFrame()
    
    import numpy as np
    
    logger.info(f"🔧 Starting format_recommendations with {len(df)} rows (skip_capping={skip_capping})")
    
    # If data is pre-capped from database, skip ALL business logic
    if skip_capping:
        logger.info("⚡ FAST PATH: Using pre-capped data from database - skipping all Python logic")
        
        # Just add required columns for output
        if 'dest_sales_used' not in df.columns:
            df['dest_sales_used'] = df.apply(
                lambda row: max(row.get('dest_wh_grn_30d_sales', 0), row.get('dest_sales', 0)),
                axis=1
            )
        
        # Only calculate missing columns (remark comes from DB)
        if 'cumulative_qty' not in df.columns:
            df['cumulative_qty'] = df.groupby(['item_code', 'dest_shop'])['recommended_qty'].cumsum()
        if 'dest_remaining_cap_before' not in df.columns:
            df['dest_remaining_cap_before'] = df['dest_sales_used'] - df['cumulative_qty'] + df['recommended_qty']
        if 'dest_updated_stock' not in df.columns:
            df['dest_updated_stock'] = df['dest_stock'] + df['recommended_qty']
        if 'dest_final_stock_days' not in df.columns:
            df['dest_final_stock_days'] = np.where(
                df['dest_sales'] > 0,
                np.ceil((df['dest_updated_stock'] / df['dest_sales']) * 30),
                999
            )
        # PRESERVE remark from mv_recommendations_complete - do NOT overwrite
        if 'remark' not in df.columns:
            df['remark'] = ''
        
        # Format output
        result = df[[
            'item_code', 'item_name', 'source_shop', 'source_stock', 'source_sales',
            'source_last_grn', 'source_grn_age', 'source_wh_grn_date', 'source_expiry_date', 'source_expiry_days',
            'expiry_check', 'expiry_status',
            'dest_shop', 'dest_stock', 'dest_sales',
            'dest_wh_grn_date', 'dest_wh_grn_plus_30', 'dest_wh_grn_30d_sales',
            'dest_sales_used', 'dest_remaining_cap_before', 'recommended_qty', 'cumulative_qty',
            'dest_updated_stock', 'dest_final_stock_days', 'remark'
        ]].copy()
        
        result.columns = [
            'ITEM_CODE', 'Item Name', 'Source Shop', 'Source Stock', 'Source Last 30d Sales',
            'Source Last GRN Date', 'Source GRN Age (days)', 'Source Last WH GRN Date', 'Source Expiry Date', 'Source Expiry Days',
            'Expiry Check', 'Expiry Status',
            'Destination Shop',
            'Destination Stock', 'Destination Last 30d Sales',
            'Destination Last WH GRN Date', 'WH GRN +30 Date', 'Destination Sales (WH GRN +30d)',
            'Destination Sales Used (Cap)', 'Destination Remaining Cap Before Allocation', 'Recommended_Qty', 'Cumulative Allocated',
            'Destination Updated Stock', 'Destination Final Stock In Hand Days', 'Remark'
        ]
        
        logger.info(f"✅ Fast path complete: {len(result)} recommendations")
        return result
    
    # ORIGINAL PYTHON LOGIC (when not using database capping)
    # Define priority shops list
    priority_shops = ['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']
    
    # Check if dest_sales_used already exists (from mv_capped_recommendations)
    if 'dest_sales_used' not in df.columns:
        # Add destination sales used column (MAX of WH GRN +30d sales or regular 30d sales)
        # CRITICAL: Calculate cap once per item+destination to ensure consistency across all source shops
        df['dest_sales_used_raw'] = df.apply(
            lambda row: max(row.get('dest_wh_grn_30d_sales', 0), row.get('dest_sales', 0)),
            axis=1
        )
        
        # Get the MAXIMUM cap value for each item+destination combination (in case of inconsistencies)
        # This ensures all source shops see the same cap for a given item+destination
        cap_lookup = df.groupby(['item_code', 'dest_shop'])['dest_sales_used_raw'].max().to_dict()
        df['dest_sales_used'] = df.apply(lambda row: cap_lookup.get((row['item_code'], row['dest_shop']), 0), axis=1)
        
        # Drop the temporary column
        df.drop(columns=['dest_sales_used_raw'], inplace=True)
        
        # Log unique cap values per destination to verify consistency
        logger.info(f"📊 Cap lookup created with {len(cap_lookup)} unique item+destination combinations")
    else:
        logger.info(f"📊 Using pre-calculated dest_sales_used from database")
    
    # DEBUG: Check dest_sales_used for item 168972 → SPN
    debug_spn = df[(df['item_code'] == '168972') & (df['dest_shop'] == 'SPN')]
    if not debug_spn.empty:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 BEFORE CAPPING - Item 168972 → SPN ({len(debug_spn)} rows)")
        logger.info(f"{'='*80}")
        unique_cap = debug_spn['dest_sales_used'].unique()
        logger.info(f"  UNIQUE dest_sales_used values: {unique_cap}")
        for idx, row in debug_spn.head(10).iterrows():
            logger.info(
                f"  Row {idx}: src={row['source_shop']:4s} | "
                f"dest_wh_grn_30d_sales={row.get('dest_wh_grn_30d_sales', 0):>6} | "
                f"dest_sales={row.get('dest_sales', 0):>6} | "
                f"dest_sales_used={row.get('dest_sales_used', 0):>6} | "
                f"recommended_qty={row.get('recommended_qty', 0):>3}"
            )
        logger.info(f"  TOTAL recommended (before cap): {debug_spn['recommended_qty'].sum()}")
        logger.info(f"{'='*80}\n")
    
    # Initialize columns
    df['skip_transfer'] = False
    df['skip_reason'] = ''
    df['remark'] = ''
    df['is_priority_dest'] = df['dest_shop'].isin(priority_shops)
    
    # RULE 1: Block if source = destination (same shop) - SKIP if using pre-capped data
    if not skip_capping:
        same_shop = df['source_shop'] == df['dest_shop']
        df.loc[same_shop, 'skip_transfer'] = True
        df.loc[same_shop, 'skip_reason'] = 'Same shop transfer'
        df.loc[same_shop, 'recommended_qty'] = 0
        
        # RULE 2: Block if source shop is a priority shop (priority shops cannot be sources)
        source_is_priority = df['source_shop'].isin(priority_shops)
        df.loc[source_is_priority, 'skip_transfer'] = True
        df.loc[source_is_priority, 'skip_reason'] = 'Source is priority shop'
        df.loc[source_is_priority, 'recommended_qty'] = 0
        
        # RULE 3: Mark "Not selling item" if all three sales are zero
        all_sales_zero = (
            (df.get('source_sales', 0) == 0) & 
            (df.get('dest_sales', 0) == 0) & 
            (df.get('dest_wh_grn_30d_sales', 0) == 0)
        )
        df.loc[all_sales_zero, 'skip_transfer'] = True
        df.loc[all_sales_zero, 'skip_reason'] = 'Not selling item'
        df.loc[all_sales_zero, 'recommended_qty'] = 0
        
        # Filter out all blocked transfers
        logger.info(f"🚫 Blocked {df['skip_transfer'].sum()} transfers")
        df = df[df['recommended_qty'] > 0].copy()
        logger.info(f"✅ Remaining {len(df)} recommendations after blocking rules")
    else:
        logger.info(f"⚡ Skipping blocking rules - data is pre-filtered from database")
    
    # Sort rows so allocation walks source shop → oldest GRN → largest qty → priority destinations
    df = df.sort_values(
        by=['source_shop', 'source_grn_age', 'recommended_qty', 'is_priority_dest', 'dest_sales_used', 'dest_shop', 'item_code'],
        ascending=[True, False, False, False, False, True, True]
    ).reset_index(drop=True)

    logger.info(
        "📊 Sorted by: source_shop ASC → source_grn_age DESC → recommended_qty DESC → "
        "priority_dest DESC → dest_sales_used DESC → dest_shop ASC → item_code ASC"
    )
    
    # DEBUG: Show sort order for item 168972
    if not df.empty and 'item_code' in df.columns:
        debug_item = df[df['item_code'] == '168972']
        if not debug_item.empty:
            logger.info(f"\n{'='*80}")
            logger.info(f"📋 SORT ORDER for Item 168972 ({len(debug_item)} rows)")
            logger.info(f"{'='*80}")
            for idx, row in debug_item.head(20).iterrows():
                logger.info(f"  idx={idx:4d} | src={row['source_shop']:4s} (age={row['source_grn_age']:3.0f}) → dest={row['dest_shop']:4s} | qty={row['recommended_qty']:3d}")
            logger.info(f"{'='*80}\n")
    
    # ===== CRITICAL CAPPING LOGIC =====
    # Skip if data is pre-capped from mv_capped_recommendations
    if skip_capping:
        logger.info("⚡ Skipping capping logic - data is pre-capped from database")
        # Just add dummy columns for compatibility
        df['cumulative_qty'] = df.groupby(['item_code', 'dest_shop'])['recommended_qty'].cumsum()
        df['capped_to_zero'] = False
        df['dest_remaining_cap_before'] = df['dest_sales_used'] - df['cumulative_qty'] + df['recommended_qty']
        df['remark'] = ''
    else:
        # Enforce per item/destination cap strictly by walking each group in sorted order
        logger.info(f"🔒 Starting capping logic on {len(df)} rows...")

        def apply_cap(group: pd.DataFrame) -> pd.DataFrame:
            # FORCE STRICT CAPPING: Get distinct cap value for this item+destination combination
            unique_caps = group['dest_sales_used'].unique()
            if len(unique_caps) > 1:
                logger.error(
                    f"❌ INCONSISTENT CAP VALUES for {group['item_code'].iloc[0]} → {group['dest_shop'].iloc[0]}: {unique_caps}. "
                    f"This should NOT happen! Using MAX value as fallback."
                )
                cap = float(max(unique_caps))
            else:
                cap = float(group['dest_sales_used'].iloc[0] or 0)
            
            item_code = group['item_code'].iloc[0]
            dest_shop = group['dest_shop'].iloc[0]
            
            logger.info(
                f"🔒 Processing {item_code} → {dest_shop}: "
                f"{len(group)} source shops, cap={cap}"
            )
            
            # Enforce strict cap - never allow total allocation > cap
            if cap <= 0:
                # No capacity - zero out all recommendations
                group['recommended_qty'] = 0
                group['cumulative_qty'] = 0
                group['capped_to_zero'] = True
                group['remark'] = 'No sales demand'
                group['dest_remaining_cap_before'] = 0
                return group
            
            allocated = 0.0
            new_qty = []
            cumulative = []
            capped_flags = []
            new_remarks = []
            remaining_cap_snapshot = []

            for _, row in group.iterrows():
                original_qty = float(row['recommended_qty'])
                src_grn_age = float(row.get('source_grn_age', 0))
                remaining_capacity = max(cap - allocated, 0)
                
                # CRITICAL: Check if cumulative sum AFTER this allocation would exceed cap
                # If yes, allocate only what fits; if no space left, allocate 0
                if allocated >= cap:
                    # Cap already reached - no more allocation possible
                    allowed_qty = 0
                else:
                    # STRICT CAP ENFORCEMENT: Never exceed remaining capacity
                    allowed_qty = min(original_qty, remaining_capacity)
                    
                    # === WEIGHTED ALLOCATION BY GRN AGE ===
                    # Apply weight to prioritize older stock (only if partial allocation needed)
                    if allowed_qty > 0 and allowed_qty < original_qty and src_grn_age > 0:
                        # Weight factor: normalize GRN age to 0-1 scale (max 365 days = full priority)
                        grn_weight = min(src_grn_age / 365.0, 1.0)  # 0 = recent, 1.0 = very old
                        weighted_qty = int(allowed_qty * grn_weight)
                        if row['item_code'] == '168972':
                            logger.info(
                                f"📊 GRN Weight: {row['item_code']} {row['source_shop']}→{row['dest_shop']} | "
                                f"age={src_grn_age}d, weight={grn_weight:.2f}, "
                                f"allowed={allowed_qty} → weighted={weighted_qty}"
                            )
                        allowed_qty = weighted_qty
                    
                    # Double-check: cumulative after this allocation must not exceed cap
                    if allocated + allowed_qty > cap:
                        logger.error(
                            f"❌ CAP VIOLATION DETECTED: {item_code} → {dest_shop} | "
                            f"allocated={allocated}, allowed_qty={allowed_qty}, cap={cap}. FORCING to cap."
                        )
                        allowed_qty = max(cap - allocated, 0)
                
                # Update cumulative AFTER determining allowed_qty
                allocated += allowed_qty

                # Log when debugging specific items
                if row['item_code'] == '168972' and row['dest_shop'] in ['SPN', 'FAR', 'LFS', 'MM2']:
                    wh_grn_sales = row.get('dest_wh_grn_30d_sales', 0)
                    regular_sales = row.get('dest_sales', 0)
                    logger.info(
                        f"🔍 Item {row['item_code']} → {row['dest_shop']} from {row['source_shop']} | "
                        f"original={original_qty}, cap={cap}, remaining_before={remaining_capacity}, allowed={allowed_qty}, cumulative_after={allocated} "
                        f"(WH GRN={wh_grn_sales}, Regular={regular_sales})"
                    )

                # Remarks based on capping result
                if allowed_qty == 0 and original_qty > 0:
                    capped_flags.append(True)
                    if row['dest_shop'] == 'SPN':
                        new_remarks.append('Any transfer will cause overstocking')
                    else:
                        new_remarks.append(f'❌ Cap reached: {int(allocated)}/{int(cap)} already allocated')
                elif allowed_qty < original_qty:
                    capped_flags.append(False)
                    new_remarks.append(f'⚠️ Partial: {int(allowed_qty)}/{int(original_qty)} (cap: {int(cap)})')
                else:
                    capped_flags.append(False)
                    new_remarks.append('')

                remaining_cap_snapshot.append(int(round(remaining_capacity)))
                new_qty.append(int(round(allowed_qty)))
                cumulative.append(int(round(allocated)))

            # FINAL VALIDATION: Ensure cumulative never exceeds cap
            final_total = sum(new_qty)
            if final_total > cap:
                logger.error(
                    f"❌ FINAL CAP VIOLATION: {item_code} → {dest_shop} | "
                    f"total_allocated={final_total} > cap={cap}. This should never happen!"
                )
                # Emergency fix: scale down all allocations proportionally
                scale_factor = cap / final_total
                new_qty = [int(round(q * scale_factor)) for q in new_qty]
                # Recalculate cumulative
                cumulative = []
                running_total = 0
                for q in new_qty:
                    running_total += q
                    cumulative.append(running_total)

            group['recommended_qty'] = new_qty
            group['cumulative_qty'] = cumulative
            group['capped_to_zero'] = capped_flags
            group['remark'] = new_remarks
            group['dest_remaining_cap_before'] = remaining_cap_snapshot
            
            # Log final allocation for verification
            if group['item_code'].iloc[0] == '168972':
                logger.info(
                    f"✅ CAPPING COMPLETE: {group['item_code'].iloc[0]} → {group['dest_shop'].iloc[0]} | "
                    f"cap={cap}, final_total={sum(new_qty)}, rows={len(group)}"
                )
            
            return group

        df = df.groupby(['item_code', 'dest_shop'], sort=False, group_keys=False).apply(apply_cap).reset_index(drop=True)
        logger.info("🔒 Capping logic completed for all item/destination groups")
    
    # CRITICAL VALIDATION: Verify no destination exceeded its cap
    if not skip_capping:
        validation = (
            df.groupby(['item_code', 'dest_shop'])
            .agg({
                'recommended_qty': 'sum',
                'dest_sales_used': 'first'
            })
            .reset_index()
        )
        validation['exceeds_cap'] = validation['recommended_qty'] > validation['dest_sales_used']
        violations = validation[validation['exceeds_cap']]
        
        if not violations.empty:
            logger.error(f"\n{'='*80}")
            logger.error(f"❌❌❌ CRITICAL: FOUND {len(violations)} CAP VIOLATIONS AFTER CAPPING!")
            logger.error(f"{'='*80}")
            for _, row in violations.iterrows():
                logger.error(
                    f"  {row['item_code']} → {row['dest_shop']}: "
                    f"total_recommended={row['recommended_qty']} > cap={row['dest_sales_used']} "
                    f"(excess: {row['recommended_qty'] - row['dest_sales_used']})"
                )
            logger.error(f"{'='*80}\n")
            
            # EMERGENCY FIX: Force scale down violations
            for _, viol in violations.iterrows():
                mask = (df['item_code'] == viol['item_code']) & (df['dest_shop'] == viol['dest_shop'])
                total = df.loc[mask, 'recommended_qty'].sum()
                cap = viol['dest_sales_used']
                if total > cap:
                    scale = cap / total
                    df.loc[mask, 'recommended_qty'] = (df.loc[mask, 'recommended_qty'] * scale).round().astype(int)
                    logger.info(f"  ✅ FIXED: Scaled {viol['item_code']} → {viol['dest_shop']} by {scale:.3f}")
        else:
            logger.info(f"✅ VALIDATION PASSED: All {len(validation)} item+destination combinations respect their caps")
    
    # DEBUG: Show final allocation for item 168972 → SPN after capping
    debug_spn_after = df[(df['item_code'] == '168972') & (df['dest_shop'] == 'SPN')]
    if not debug_spn_after.empty:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 AFTER CAPPING - Item 168972 → SPN ({len(debug_spn_after)} rows)")
        logger.info(f"{'='*80}")
        total_recommended = 0
        for idx, row in debug_spn_after.iterrows():
            total_recommended += row.get('recommended_qty', 0)
            logger.info(
                f"  Row {idx}: src={row['source_shop']:4s} (age={row.get('source_grn_age', 0):>3.0f}d) | "
                f"cap={row.get('dest_sales_used', 0):>6} | "
                f"remaining_before={row.get('dest_remaining_cap_before', 0):>6} | "
                f"recommended_qty={row.get('recommended_qty', 0):>3} | "
                f"cumulative={row.get('cumulative_qty', 0):>6} | "
                f"remark={row.get('remark', '')}"
            )
        logger.info(f"  TOTAL recommended (after cap): {total_recommended}")
        logger.info(f"  CAP value: {debug_spn_after['dest_sales_used'].iloc[0]}")
        cap_value = debug_spn_after['dest_sales_used'].iloc[0]
        if total_recommended > cap_value:
            logger.error(f"❌❌❌ CRITICAL: Total {total_recommended} EXCEEDS cap {cap_value} by {total_recommended - cap_value}!")
        else:
            logger.info(f"✅ PASS: Total {total_recommended} ≤ cap {cap_value}")
        logger.info(f"{'='*80}\n")
    
    
    # Set remarks for non-capped transfers (only if remark is empty)
    df.loc[(df['recommended_qty'] > 0) & (df['remark'] == '') & df['is_priority_dest'], 'remark'] = 'Priority transfer'
    df.loc[(df['recommended_qty'] > 0) & (df['remark'] == '') & ~df['is_priority_dest'], 'remark'] = 'Normal transfer'
    
    # Log summary
    capped_count = df['capped_to_zero'].sum()
    logger.info(f"🔒 Capping complete: {capped_count} transfers capped to 0")
    
    # DEBUG: Show final allocation for item 168972
    item_168972 = df[df['item_code'] == '168972']
    if not item_168972.empty:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 FINAL ALLOCATION SUMMARY - Item 168972")
        logger.info(f"{'='*80}")
        summary = (
            item_168972.groupby('dest_shop')[['recommended_qty', 'dest_sales_used']]
            .agg({'recommended_qty': 'sum', 'dest_sales_used': 'first'})
            .reset_index()
        )
        violations = []
        for _, row in summary.iterrows():
            total_rec = row['recommended_qty']
            cap = row['dest_sales_used']
            status = "✅ OK" if total_rec <= cap else f"❌ VIOLATION ({total_rec - cap} over)"
            logger.info(
                f"  DEST={row['dest_shop']}: allocated={total_rec} vs cap={cap} {status}"
            )
            if total_rec > cap:
                violations.append((row['dest_shop'], total_rec, cap))
        
        if violations:
            logger.error(f"\n❌❌❌ FOUND {len(violations)} CAP VIOLATIONS:")
            for shop, total, cap in violations:
                logger.error(f"  {shop}: {total} recommended but cap is {cap}")
        else:
            logger.info(f"✅ All destinations respect caps!")
        logger.info(f"{'='*80}\n")
    
    # Calculate final metrics
    df['dest_updated_stock'] = df['dest_stock'] + df['recommended_qty']
    df['dest_final_stock_days'] = np.where(
        df['dest_sales'] > 0,
        np.ceil((df['dest_updated_stock'] / df['dest_sales']) * 30),
        np.where(
            df['dest_updated_stock'] > 0,
            999,  # Has stock but no sales
            0     # No stock
        )
    )
    
    logger.info(f"✅ Format complete: {len(df)} final recommendations")
    
    # DON'T remove rows that were capped - keep them with qty=0 and remark
    # Only filter was already done for skip_transfer rows
    
    # Recalculate final metrics after capping
    df['dest_updated_stock'] = df['dest_stock'] + df['recommended_qty']
    
    # Calculate destination final stock in hand days
    # Formula: (dest_updated_stock / dest_sales) * 30, rounded up
    df['dest_final_stock_days'] = np.where(
        df['dest_updated_stock'] == 0,
        0,
        np.where(
            (df['dest_stock'] == 0) & (df['dest_sales'] == 0) & (df['recommended_qty'] > 0),
            df['recommended_qty'],  # If dest has no stock and no sales, show recommended qty
            np.where(
                df['dest_sales'] > 0,
                np.ceil((df['dest_updated_stock'] / df['dest_sales']) * 30),
                np.where(
                    (df['dest_stock'] > 0) & (df['dest_sales'] == 0) & (df['recommended_qty'] > 0),
                    df['dest_updated_stock'],  # If dest has stock but no sales, show updated stock
                    999
                )
            )
        )
    )
    
    # STEP 4: DEBUG - Check item 168972 BEFORE selecting output columns
    if 'item_code' in df.columns:
        debug_item = df[df['item_code'] == '168972']
        problem_shops = ['SPN', 'LFS', 'FAR', 'MM2', 'KAS']
        if not debug_item.empty:
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 4: BEFORE OUTPUT COLUMN SELECTION - Item 168972 ({len(debug_item)} total rows)")
            logger.info(f"{'='*80}")
            for shop in problem_shops:
                shop_rows = debug_item[debug_item['dest_shop'] == shop]
                if not shop_rows.empty:
                    logger.info(f"\n  DEST={shop}: {len(shop_rows)} rows")
                    for idx, row in shop_rows.iterrows():
                        logger.info(f"    Row {idx}: src={row['source_shop']:4s} | dest_wh_grn_30d_sales={row.get('dest_wh_grn_30d_sales', 'N/A'):>6} | rec_qty={row.get('recommended_qty', 'N/A'):>3} | skip={row.get('skip_transfer', 'N/A')}")
                else:
                    logger.info(f"\n  DEST={shop}: FILTERED OUT (skip_transfer or recommended_qty=0)")
            logger.info(f"{'='*80}\n")
    
    # Format output columns - INCLUDE dest_sales_used and cumulative_qty for visibility
    result = df[[
        'item_code', 'item_name', 'source_shop', 'source_stock', 'source_sales',
        'source_last_grn', 'source_grn_age', 'source_wh_grn_date', 'dest_shop', 'dest_stock', 'dest_sales',
        'dest_wh_grn_date', 'dest_wh_grn_plus_30', 'dest_wh_grn_30d_sales',
        'dest_sales_used', 'dest_remaining_cap_before', 'recommended_qty', 'cumulative_qty',
        'dest_updated_stock', 'dest_final_stock_days', 'remark'
    ]].copy()
    
    result.columns = [
        'ITEM_CODE', 'Item Name', 'Source Shop', 'Source Stock', 'Source Last 30d Sales',
        'Source Last GRN Date', 'Source GRN Age (days)', 'Source Last WH GRN Date', 'Destination Shop',
        'Destination Stock', 'Destination Last 30d Sales',
        'Destination Last WH GRN Date', 'WH GRN +30 Date', 'Destination Sales (WH GRN +30d)',
        'Destination Sales Used (Cap)', 'Destination Remaining Cap Before Allocation', 'Recommended_Qty', 'Cumulative Allocated', 
        'Destination Updated Stock', 'Destination Final Stock In Hand Days', 'Remark'
    ]
    
    # Format GRN dates
    result['Source Last GRN Date'] = result['Source Last GRN Date'].fillna('N/A')
    result['Source Last WH GRN Date'] = result['Source Last WH GRN Date'].fillna('N/A')
    
    # WH GRN date is item-level, so source and destination should always be the same
    # If destination WH GRN date is missing or N/A, use source WH GRN date
    result['Destination Last WH GRN Date'] = result.apply(
        lambda row: row['Source Last WH GRN Date'] 
        if (pd.isna(row['Destination Last WH GRN Date']) or 
            row['Destination Last WH GRN Date'] == 'N/A' or 
            row['Destination Last WH GRN Date'] == '')
        else row['Destination Last WH GRN Date'],
        axis=1
    )
    
    # Recalculate WH GRN +30 Date based on Destination Last WH GRN Date
    # If destination WH GRN date is valid, calculate +30 days, otherwise N/A
    def calculate_grn_plus_30(wh_grn_date):
        if pd.notna(wh_grn_date) and wh_grn_date != 'N/A' and wh_grn_date != '':
            try:
                if isinstance(wh_grn_date, str):
                    date_obj = pd.to_datetime(wh_grn_date)
                else:
                    date_obj = pd.Timestamp(wh_grn_date)
                return (date_obj + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            except:
                return 'N/A'
        return 'N/A'
    
    result['WH GRN +30 Date'] = result['Destination Last WH GRN Date'].apply(calculate_grn_plus_30)
    
    # Ensure Destination Sales (WH GRN +30d) is numeric and properly formatted
    result['Destination Sales (WH GRN +30d)'] = pd.to_numeric(
        result['Destination Sales (WH GRN +30d)'], errors='coerce'
    ).fillna(0).astype(int)
    
    # Show destination-level columns only ONCE per item-destination group (not repeated for each source)
    # Mark duplicates within each (ITEM_CODE, Destination Shop) group
    result['_is_first_in_group'] = ~result.duplicated(subset=['ITEM_CODE', 'Destination Shop'], keep='first')
    
    # Blank out repeated destination-level values (keep only first occurrence)
    destination_columns_to_blank = [
        'Destination Stock',
        'Destination Last 30d Sales', 
        'Destination Last WH GRN Date',
        'WH GRN +30 Date',
        'Destination Sales (WH GRN +30d)',
        'Destination Sales Used (Cap)'
    ]
    
    for col in destination_columns_to_blank:
        if col in result.columns:
            # Replace with empty string for non-first rows in each group
            result.loc[~result['_is_first_in_group'], col] = ''
    
    # Remove the helper column
    result = result.drop(columns=['_is_first_in_group'])
    
    # FINAL SAFETY: Remove any duplicate rows before returning
    initial_count = len(result)
    result = result.drop_duplicates().reset_index(drop=True)
    if len(result) < initial_count:
        logger.warning(f"⚠️ FINAL DEDUP: Removed {initial_count - len(result)} duplicate rows from output")
    
    # STEP 5: DEBUG - Check item 168972 in FINAL OUTPUT
    if 'ITEM_CODE' in result.columns:
        debug_item = result[result['ITEM_CODE'] == '168972']
        problem_shops = ['SPN', 'LFS', 'FAR', 'MM2', 'KAS']
        if not debug_item.empty:
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 5: FINAL OUTPUT - Item 168972 ({len(debug_item)} total rows)")
            logger.info(f"{'='*80}")
            for shop in problem_shops:
                shop_rows = debug_item[debug_item['Destination Shop'] == shop]
                if not shop_rows.empty:
                    logger.info(f"\n  DEST={shop}: {len(shop_rows)} rows in final output")
                    for idx, row in shop_rows.iterrows():
                        logger.info(f"    Row {idx}: src={row['Source Shop']:4s} | Destination Sales (WH GRN +30d)={row['Destination Sales (WH GRN +30d)']:>6} | Recommended_Qty={row['Recommended_Qty']:>3}")
                else:
                    logger.info(f"\n  DEST={shop}: NOT IN FINAL OUTPUT")
            logger.info(f"{'='*80}\n")
    
    return result


def generate_recommendations_optimized(group: str = 'All', subgroup: str = 'All', 
                                      product: str = 'All', shop: str = 'All',
                                      use_grn_logic: bool = True, threshold: int = 10,
                                      use_cache: bool = True, limit: int = None) -> pd.DataFrame:
    """
    SIMPLIFIED: Query mv_recommendations_complete with filters only.
    ALL business logic is in the database view - Python just filters and displays.
    
    Performance: <100ms (view contains pre-computed results)
    """
    import time
    
    start_time = time.time()
    logger.info(f"🚀 Querying mv_recommendations_complete")
    
    # Build WHERE clause for filters
    where_conditions = []
    params = []
    
    if group and group != 'All':
        where_conditions.append("groups = %s")
        params.append(group)
    if subgroup and subgroup != 'All':
        where_conditions.append("sub_group = %s")
        params.append(subgroup)
    if product and product != 'All':
        where_conditions.append("item_code = %s")
        params.append(product.strip().upper())
    if shop and shop != 'All':
        where_conditions.append("(source_shop = %s OR dest_shop = %s)")
        params.extend([shop.strip().upper(), shop.strip().upper()])
    
    where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
    
    query = f"""
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
            expiry_status,
            dest_shop,
            dest_stock,
            dest_sales,
            dest_wh_grn_date,
            dest_wh_grn_plus_30,
            dest_wh_grn_30d_sales,
            dest_sales_used,
            dest_remaining_cap_before,
            recommended_qty,
            cumulative_qty,
            dest_updated_stock,
            dest_final_stock_days,
            remark
        FROM mv_recommendations_complete
        {where_clause}
        ORDER BY source_grn_age DESC, priority_rank ASC, dest_shop, item_code
        {f'LIMIT {limit}' if limit else ''}
    """
    
    try:
        with get_db_connection('salesdata') as conn:
            if params:
                df = pd.read_sql(query, conn, params=params)
            else:
                df = pd.read_sql(query, conn)
        
        if df.empty:
            logger.warning("⚠️ No recommendations found with current filters")
            return pd.DataFrame()
        
        query_time = time.time() - start_time
        logger.info(f"⚡ Loaded {len(df)} recommendations in {query_time:.2f}s")
        
        # Format output (just column names and display formatting)
        result = format_recommendations(df, skip_capping=True)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Complete in {total_time:.2f}s | {len(result)} recommendations")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error querying mv_recommendations_complete: {e}")
        st.error(f"❌ Database view not available.\n\nPlease run:\n1. cd d:\\Dashboard Code\\NO_WH\\DS\n2. psql -U postgres -d salesdata -p 3307 -f run_this_first.sql")
        return pd.DataFrame()


def generate_recommendations(df: pd.DataFrame, use_grn_logic: bool = True, threshold: int = 10) -> pd.DataFrame:
    """
    ENHANCED recommendation engine with updated business logic.
    
    NEW BUSINESS RULES (Nov 2025):
    1. ONLY priority shops as destinations (11 shops: SPN, MSS, LFS, M03, KAS, MM1, MM2, FAR, KS7, WHL, MM3)
    2. Calculate "Updated Last 30d Sales" = sales from (latest WH GRN date) to (GRN date + 30 days)
    3. Recommended Qty = MAX(Last 30d Sales, Updated Last 30d Sales) - NO 30-UNIT CAP
    4. Sum of recommended qty per source shop cannot exceed source shop's available stock
    5. Prioritize based on which sales figure is higher
    
    Performance: Optimized with vectorized operations where possible
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import time

    if df is None or df.empty:
        logger.warning("Empty dataframe passed to generate_recommendations")
        return pd.DataFrame()

    start_time = time.time()
    logger.info(f"🚀 NEW LOGIC | Generating recommendations (GRN: {use_grn_logic}) | Input rows: {len(df)}")

    # --- normalize input data ---
    base_df = df.copy()
    base_df['ITEM_CODE'] = base_df['ITEM_CODE'].str.strip().str.upper()
    base_df['SHOP_CODE'] = base_df['SHOP_CODE'].str.strip().str.upper()
    
    # Ensure GROUPS and SUB_GROUP columns exist
    if 'GROUPS' not in base_df.columns:
        logger.warning("GROUPS column missing from input data")
        base_df['GROUPS'] = 'N/A'
    if 'SUB_GROUP' not in base_df.columns:
        logger.warning("SUB_GROUP column missing from input data")
        base_df['SUB_GROUP'] = 'N/A'

    # --- Load standard 30-day sales ---
    sales_30d_df = load_last_30d_sales()
    if not sales_30d_df.empty:
        sales_30d_df.columns = sales_30d_df.columns.str.upper()
        sales_30d_df['ITEM_CODE'] = sales_30d_df['ITEM_CODE'].str.strip().str.upper()
        sales_30d_df['SHOP_CODE'] = sales_30d_df['SHOP_CODE'].str.strip().str.upper()
        base_df = base_df.merge(sales_30d_df, on=['ITEM_CODE', 'SHOP_CODE'], how='left')
        base_df['LAST_30D_SALES'] = base_df.get('SALES_30D', base_df.get('ITEM_SALES_30_DAYS', 0)).fillna(0).astype(float)
        base_df = base_df.drop(columns=['SALES_30D', 'ITEM_SALES_30_DAYS'], errors='ignore')
    else:
        base_df['LAST_30D_SALES'] = base_df.get('ITEM_SALES_30_DAYS', 0).fillna(0).astype(float)

    # --- Load UPDATED Last 30d Sales (from WH GRN date + 30 days) ---
    # This is the NEW business requirement
    # OPTIMIZATION: Try to get from inventory_master first (faster), fallback to calculation
    if 'WH_GRN_SALES_30D' in base_df.columns:
        # Already loaded from inventory_master - FASTEST
        base_df['UPDATED_LAST_30D_SALES'] = base_df['WH_GRN_SALES_30D'].fillna(0).astype(float)
        logger.info(f"✅ Using pre-computed WH GRN sales from inventory_master (FASTEST)")
    else:
        # Fallback: Calculate on-the-fly
        shop_grn_sales = load_shop_grn_sales()  # Load for all shops
        if not shop_grn_sales.empty:
            shop_grn_sales['ITEM_CODE'] = shop_grn_sales['ITEM_CODE'].str.strip().str.upper()
            shop_grn_sales['SHOP_CODE'] = shop_grn_sales['SHOP_CODE'].str.strip().str.upper()
            base_df = base_df.merge(shop_grn_sales, on=['ITEM_CODE', 'SHOP_CODE'], how='left')
            base_df['UPDATED_LAST_30D_SALES'] = base_df.get('SHOP_GRN_SALES_30D', 0).fillna(0).astype(float)
            logger.info(f"✅ Calculated Updated Last 30d Sales on-the-fly")
        else:
            base_df['UPDATED_LAST_30D_SALES'] = 0

    # --- Load GRN dates from sup_shop_grn table (in salesdata db) ---
    try:
        with get_db_connection('salesdata') as conn:
            # Load shop GRN dates (per item-shop)
            shop_grn_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER(item_code)) AS item_code,
                    TRIM(UPPER(shop_code)) AS shop_code,
                    MAX(shop_grn_date) AS shop_grn_date
                FROM sup_shop_grn
                WHERE shop_grn_date IS NOT NULL
                GROUP BY 1, 2
            """, conn)
            
            # CRITICAL: Load WH GRN dates (per ITEM ONLY - warehouse level)
            wh_grn_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER(item_code)) AS item_code,
                    MAX(wh_grn_date) AS wh_grn_date
                FROM sup_shop_grn
                WHERE wh_grn_date IS NOT NULL
                GROUP BY 1
            """, conn)
            
            # Merge shop GRN dates
            if not shop_grn_df.empty:
                shop_grn_df.columns = ['ITEM_CODE', 'SHOP_CODE', 'SHOP_GRN_DATE_FROM_GRN']
                base_df = base_df.merge(shop_grn_df, on=['ITEM_CODE', 'SHOP_CODE'], how='left')
                base_df['SHOP_GRN_DATE'] = base_df['SHOP_GRN_DATE_FROM_GRN'].fillna(base_df.get('SHOP_GRN_DATE', pd.NaT))
                base_df = base_df.drop(columns=['SHOP_GRN_DATE_FROM_GRN'], errors='ignore')
            
            # Merge WH GRN dates (ITEM-LEVEL - same for all shops)
            # CRITICAL: Only fill missing values, don't overwrite existing WH_GRN_DATE from inventory_master
            if not wh_grn_df.empty:
                wh_grn_df.columns = ['ITEM_CODE', 'WH_GRN_DATE_FROM_GRN']
                base_df = base_df.merge(wh_grn_df, on='ITEM_CODE', how='left')
                # Use existing WH_GRN_DATE from inventory_master if available, otherwise use from sup_shop_grn
                if 'WH_GRN_DATE' in base_df.columns:
                    base_df['WH_GRN_DATE'] = base_df['WH_GRN_DATE'].fillna(base_df['WH_GRN_DATE_FROM_GRN'])
                else:
                    base_df['WH_GRN_DATE'] = base_df['WH_GRN_DATE_FROM_GRN']
                base_df = base_df.drop(columns=['WH_GRN_DATE_FROM_GRN'], errors='ignore')
                logger.info(f"✅ Loaded WH GRN dates (item-level) for {len(wh_grn_df)} unique items")
            
            if not shop_grn_df.empty:
                logger.info(f"✅ Loaded shop GRN dates for {len(shop_grn_df)} item-shop combinations")
                
    except Exception as e:
        logger.warning(f"Could not load GRN dates from sup_shop_grn: {e}")
        if 'SHOP_GRN_DATE' not in base_df.columns:
            base_df['SHOP_GRN_DATE'] = pd.NaT
        if 'WH_GRN_DATE' not in base_df.columns:
            base_df['WH_GRN_DATE'] = pd.NaT

    # Convert dates to datetime
    base_df['SHOP_GRN_DATE'] = pd.to_datetime(base_df.get('SHOP_GRN_DATE', pd.NaT), errors='coerce')
    
    # CRITICAL: Ensure WH_GRN_DATE exists (loaded from inventory_master as whgrn_dt)
    if 'WH_GRN_DATE' not in base_df.columns:
        logger.warning("WH_GRN_DATE not in base_df columns, loading from sup_shop_grn")
        # Will be loaded from sup_shop_grn below
        base_df['WH_GRN_DATE'] = pd.NaT
    
    base_df['WH_GRN_DATE'] = pd.to_datetime(base_df['WH_GRN_DATE'], errors='coerce')
    base_df['GRN_DATE'] = base_df['SHOP_GRN_DATE']
    
    # --- CRITICAL: Ensure ALL priority shops appear as potential destinations ---
    all_items = base_df['ITEM_CODE'].unique()
    all_priority_shops_idx = pd.MultiIndex.from_product(
        [all_items, Config.PRIORITY_SHOPS], names=['ITEM_CODE', 'SHOP_CODE']
    )
    missing = all_priority_shops_idx.difference(pd.MultiIndex.from_frame(base_df[['ITEM_CODE', 'SHOP_CODE']]))
    
    if len(missing) > 0:
        mdf = pd.DataFrame(list(missing), columns=['ITEM_CODE', 'SHOP_CODE'])
        mdf['ITEM_NAME'] = mdf['ITEM_CODE'].map(
            base_df.drop_duplicates('ITEM_CODE').set_index('ITEM_CODE')['ITEM_NAME'].to_dict()
        )
        mdf['SHOP_STOCK'] = 0
        mdf['LAST_30D_SALES'] = 0
        mdf['UPDATED_LAST_30D_SALES'] = 0
        mdf['SHOP_GRN_DATE'] = pd.NaT
        mdf['WH_GRN_DATE'] = pd.NaT
        mdf['GRN_DATE'] = pd.NaT
        base_df = pd.concat([base_df, mdf], ignore_index=True)
        logger.info(f"✅ Added {len(missing)} missing priority shop-item combinations")

    # --- Filter to ONLY priority shops as destinations ---
    priority_df = base_df[base_df['SHOP_CODE'].isin(Config.PRIORITY_SHOPS)].copy()
    
    if priority_df.empty:
        logger.warning("No priority shops found in base_df")
        return pd.DataFrame()
    
    logger.info(f"✅ Filtered to {len(priority_df)} priority shop records (destinations only)")
    
    # Debug: Check GRN data in base_df before merge
    logger.info(f"Columns in base_df: {list(base_df.columns)}")
    non_null_shop_grn = base_df['SHOP_GRN_DATE'].notna().sum()
    non_null_wh_grn = base_df['WH_GRN_DATE'].notna().sum() if 'WH_GRN_DATE' in base_df.columns else 0
    non_null_updated_sales = (base_df['UPDATED_LAST_30D_SALES'] > 0).sum() if 'UPDATED_LAST_30D_SALES' in base_df.columns else 0
    logger.info(f"Before merge - Non-null SHOP_GRN_DATE: {non_null_shop_grn}, WH_GRN_DATE: {non_null_wh_grn}, UPDATED_LAST_30D_SALES>0: {non_null_updated_sales}")
    
    # --- Merge: ALL sources × PRIORITY destinations ---
    merged = base_df.merge(priority_df, on='ITEM_CODE', suffixes=('_SRC', '_DST'), how='inner')
    
    # Debug: Check if GRN dates are in merged dataframe
    grn_cols = [col for col in merged.columns if 'GRN' in col.upper()]
    logger.info(f"GRN-related columns in merged data: {grn_cols}")
    if 'SHOP_GRN_DATE_DST' in merged.columns:
        non_null_grn = merged['SHOP_GRN_DATE_DST'].notna().sum()
        logger.info(f"Non-null SHOP_GRN_DATE_DST values: {non_null_grn} out of {len(merged)}")
    
    # Double-check destinations are ONLY priority shops
    merged = merged[merged['SHOP_CODE_DST'].isin(Config.PRIORITY_SHOPS)]
    
    # Filter source shops with stock > 30 (can transfer surplus)
    merged = merged[merged['SHOP_STOCK_SRC'] > 30]
    
    # Remove same-shop transfers
    merged = merged[merged['SHOP_CODE_SRC'] != merged['SHOP_CODE_DST']]
    
    if merged.empty:
        logger.warning("No valid shop combinations found")
        return pd.DataFrame()
    
    logger.info(f"⚡ Processing {len(merged)} shop-to-shop combinations")

    recs = []
    today = pd.Timestamp.today().date()
    
    # Track allocations per source shop per item to enforce stock limits
    source_allocations = {}  # Key: (item, source_shop), Value: total allocated

    for (item, src_shop), g in merged.groupby(['ITEM_CODE', 'SHOP_CODE_SRC'], observed=True):
        src_row = g.iloc[0]
        src_stock = int(src_row['SHOP_STOCK_SRC'] or 0)
        src_sales = float(src_row.get('LAST_30D_SALES_SRC', 0) or 0)
        item_name = src_row.get('ITEM_NAME_SRC', item)
        
        # Get Group and Sub Group
        item_group = src_row.get('GROUPS_SRC', src_row.get('GROUPS', 'N/A'))
        item_subgroup = src_row.get('SUB_GROUP_SRC', src_row.get('SUB_GROUP', 'N/A'))

        # Calculate available stock from source
        src_needed_30d = src_sales if src_sales > 0 else 30
        available = max(src_stock - src_needed_30d, 0)
        
        if available <= 0:
            continue  # Skip sources with no available stock

        # Get source GRN info (both shop and WH)
        src_last_grn = src_row.get('SHOP_GRN_DATE_SRC', src_row.get('GRN_DATE_SRC', pd.NaT))
        src_wh_grn = src_row.get('WH_GRN_DATE_SRC', pd.NaT)
        src_grn_age_days = 0
        src_grn_is_recent = False
        
        if pd.notna(src_last_grn):
            src_grn_date = pd.Timestamp(src_last_grn)
            if src_grn_date.tz is not None:
                src_grn_date = src_grn_date.tz_localize(None)
            src_grn_age_days = (pd.Timestamp(today) - src_grn_date).days
            if src_grn_age_days < 30:
                src_grn_is_recent = True

        # Skip if source GRN is recent (within 30 days)
        if use_grn_logic and src_grn_is_recent:
            continue

        # Initialize source allocation tracker
        source_key = (item, src_shop)
        if source_key not in source_allocations:
            source_allocations[source_key] = 0

        # Process all priority destinations for this item
        destination_recommendations = []
        
        for dest_shop in Config.PRIORITY_SHOPS:
            if dest_shop == src_shop:
                continue  # Skip same-shop transfers

            # Get destination data
            drow = g[g['SHOP_CODE_DST'] == dest_shop]
            
            if not drow.empty:
                dest_stock = int(drow['SHOP_STOCK_DST'].iloc[0] or 0)
                dest_last_30d = float(drow['LAST_30D_SALES_DST'].iloc[0] or 0)
                dest_updated_30d = float(drow.get('UPDATED_LAST_30D_SALES_DST', pd.Series([0])).iloc[0] or 0)
                # Get destination GRN dates (both shop and WH)
                dest_last_grn = drow['SHOP_GRN_DATE_DST'].iloc[0] if 'SHOP_GRN_DATE_DST' in drow.columns else pd.NaT
                dest_wh_grn = drow['WH_GRN_DATE_DST'].iloc[0] if 'WH_GRN_DATE_DST' in drow.columns else pd.NaT
            else:
                # Look up from base_df
                dest_lookup = base_df[(base_df['ITEM_CODE'] == item) & (base_df['SHOP_CODE'] == dest_shop)]
                if not dest_lookup.empty:
                    dest_stock = int(dest_lookup['SHOP_STOCK'].iloc[0] or 0)
                    dest_last_30d = float(dest_lookup.get('LAST_30D_SALES', 0).iloc[0] or 0)
                    dest_updated_30d = float(dest_lookup.get('UPDATED_LAST_30D_SALES', 0).iloc[0] or 0)
                    # Try multiple columns for GRN dates
                    dest_last_grn = dest_lookup.get('SHOP_GRN_DATE', dest_lookup.get('GRN_DATE', pd.Series([pd.NaT]))).iloc[0]
                    dest_wh_grn = dest_lookup.get('WH_GRN_DATE', pd.Series([pd.NaT])).iloc[0]
                else:
                    dest_stock = 0
                    dest_last_30d = 0
                    dest_updated_30d = 0
                    dest_last_grn = pd.NaT
                    dest_wh_grn = pd.NaT

            # Calculate destination GRN age
            dest_grn_age_days = 0
            if pd.notna(dest_last_grn):
                dest_grn_date = pd.Timestamp(dest_last_grn)
                if dest_grn_date.tz is not None:
                    dest_grn_date = dest_grn_date.tz_localize(None)
                dest_grn_age_days = (pd.Timestamp(today) - dest_grn_date).days

            # === NEW LOGIC: Use MAX of Last 30d Sales and Updated Last 30d Sales ===
            dest_sales = max(dest_last_30d, dest_updated_30d)
            sales_source_used = "Last 30d" if dest_last_30d >= dest_updated_30d else "Updated 30d"
            
            # Calculate base recommended quantity (NO CAP AT ALL)
            if dest_sales > 0:
                # Sales-based calculation - use MAX(Last 30d, Updated 30d)
                if dest_stock == 0:
                    base_recommended_qty = dest_sales
                else:
                    current_stock_days = (dest_stock / dest_sales) * 30
                    if current_stock_days < 30:
                        base_recommended_qty = max(0, dest_sales - dest_stock)
                    else:
                        base_recommended_qty = 0  # Already has 30+ days stock
            else:
                # No sales at all - don't recommend anything
                base_recommended_qty = 0

            base_recommended_qty = max(0, int(base_recommended_qty))
            
            # Store ALL priority destinations (even with 0 qty) to show complete picture
            destination_recommendations.append({
                'dest_shop': dest_shop,
                'dest_stock': dest_stock,
                'dest_last_30d': dest_last_30d,
                'dest_updated_30d': dest_updated_30d,
                'dest_wh_grn_plus_30': pd.Timestamp(dest_wh_grn) + pd.Timedelta(days=30) if pd.notna(dest_wh_grn) else pd.NaT,
                'dest_sales': dest_sales,
                'sales_source': sales_source_used,
                'dest_last_grn': dest_last_grn,
                'dest_wh_grn': dest_wh_grn,
                'dest_grn_age': dest_grn_age_days,
                'base_qty': base_recommended_qty,
                'grn_weight': min(src_grn_age_days / 365.0, 1.0)  # Older GRN = higher priority
            })

        # Sort destinations by GRN age (prioritize shops with older source GRN)
        destination_recommendations.sort(key=lambda x: x['grn_weight'], reverse=True)

        # Allocate available stock to destinations
        remaining_available = available - source_allocations[source_key]
        
        for dest_rec in destination_recommendations:
            # Allocate quantity (capped by remaining available stock)
            if remaining_available > 0 and dest_rec['base_qty'] > 0:
                allocated_qty = min(dest_rec['base_qty'], remaining_available)
                # Update trackers
                source_allocations[source_key] += allocated_qty
                remaining_available -= allocated_qty
            else:
                allocated_qty = 0

            # Calculate final metrics
            dest_updated_stock = dest_rec['dest_stock'] + allocated_qty
            if dest_rec['dest_sales'] > 0:
                final_stock_days = (dest_updated_stock / dest_rec['dest_sales']) * 30
            else:
                final_stock_days = 0

            # Convert GRN dates for export
            src_grn_export = src_last_grn
            if pd.notna(src_last_grn):
                src_grn_ts = pd.Timestamp(src_last_grn)
                if src_grn_ts.tz is not None:
                    src_grn_export = src_grn_ts.tz_localize(None)
            
            src_wh_grn_export = src_wh_grn
            if pd.notna(src_wh_grn):
                src_wh_grn_ts = pd.Timestamp(src_wh_grn)
                if src_wh_grn_ts.tz is not None:
                    src_wh_grn_export = src_wh_grn_ts.tz_localize(None)
            
            dest_grn_export = dest_rec['dest_last_grn']
            if pd.notna(dest_rec['dest_last_grn']):
                dest_grn_ts = pd.Timestamp(dest_rec['dest_last_grn'])
                if dest_grn_ts.tz is not None:
                    dest_grn_export = dest_grn_ts.tz_localize(None)

            # Determine slow/fast moving status for source only
            src_slow_moving = 'Yes' if src_sales < threshold else 'No'
            src_fast_moving = 'Yes' if src_sales >= threshold else 'No'
            
            # Create remark based on allocated quantity
            if allocated_qty > 0:
                remark = f'✅ Priority transfer ({dest_rec["sales_source"]})'
            elif dest_rec['base_qty'] > 0:
                remark = '⚠️ No stock available from source'
            else:
                remark = 'ℹ️ No transfer needed (sufficient stock)'
            
            # Create recommendation record
            recs.append({
                'ITEM_CODE': item,
                'Item Name': item_name,
                'Group': item_group,
                'Sub Group': item_subgroup,
                'Source Shop': src_shop,
                'Source Stock': src_stock,
                'Source Last 30d Sales': src_sales,
                'Source Slow Moving': src_slow_moving,
                'Source Fast Moving': src_fast_moving,
                'Source Last GRN Date': src_grn_export if pd.notna(src_grn_export) else 'N/A',
                'Source Last WH GRN Date': src_wh_grn_export if pd.notna(src_wh_grn_export) else 'N/A',
                'Source GRN Age (days)': src_grn_age_days,
                'Destination Shop': dest_rec['dest_shop'],
                'Destination Stock': dest_rec['dest_stock'],
                'Destination Last 30d Sales': dest_rec['dest_last_30d'],
                'Destination Last WH GRN Date': src_wh_grn_export if pd.notna(src_wh_grn_export) else 'N/A',
                'WH GRN +30 Date': (pd.Timestamp(src_wh_grn_export) + pd.Timedelta(days=30)).strftime('%Y-%m-%d') if pd.notna(src_wh_grn_export) else 'N/A',
                'Destination Sales (WH GRN +30d)': dest_rec['dest_updated_30d'],
                'Destination Sales Used': dest_rec['dest_sales'],
                'Sales Source': dest_rec['sales_source'],
                'Destination Last GRN Date': dest_grn_export if pd.notna(dest_grn_export) else 'N/A',
                'Destination GRN Age (days)': dest_rec['dest_grn_age'],
                'Recommended_Qty': allocated_qty,
                'Destination Updated Stock': dest_updated_stock,
                'Destination Final Stock In Hand Days': np.round(final_stock_days, 1),
                'Remark': remark
            })

    # Convert to DataFrame
    if not recs:
        logger.warning("No recommendations generated")
        return pd.DataFrame()

    result = pd.DataFrame(recs)
    
    # Ensure ALL items have ALL 11 priority shops represented
    all_items_in_result = result['ITEM_CODE'].unique()
    all_priority_combinations = pd.MultiIndex.from_product(
        [all_items_in_result, Config.PRIORITY_SHOPS],
        names=['ITEM_CODE', 'Destination Shop']
    )
    existing_combinations = pd.MultiIndex.from_frame(
        result[['ITEM_CODE', 'Destination Shop']]
    )
    missing_combinations = all_priority_combinations.difference(existing_combinations)
    
    if len(missing_combinations) > 0:
        logger.info(f"📝 Adding {len(missing_combinations)} missing item-destination combinations (all 11 priority shops)")
        
        # Create records for missing combinations
        missing_recs = []
        for item_code, dest_shop in missing_combinations:
            # Get item info from base_df
            item_info = base_df[base_df['ITEM_CODE'] == item_code].iloc[0] if not base_df[base_df['ITEM_CODE'] == item_code].empty else None
            
            if item_info is not None:
                item_name = item_info.get('ITEM_NAME', 'Unknown')
                item_group = item_info.get('GROUPS', 'N/A')
                item_subgroup = item_info.get('SUB_GROUP', 'N/A')
                
                # Get destination info
                dest_info = base_df[(base_df['ITEM_CODE'] == item_code) & (base_df['SHOP_CODE'] == dest_shop)]
                if not dest_info.empty:
                    dest_stock = int(dest_info['SHOP_STOCK'].iloc[0] or 0)
                    dest_last_30d = float(dest_info.get('LAST_30D_SALES', 0).iloc[0] or 0)
                    dest_updated_30d = float(dest_info.get('UPDATED_LAST_30D_SALES', 0).iloc[0] or 0)
                    dest_last_grn = dest_info.get('SHOP_GRN_DATE', pd.Series([pd.NaT])).iloc[0]
                    dest_wh_grn = dest_info.get('WH_GRN_DATE', pd.Series([pd.NaT])).iloc[0]
                else:
                    dest_stock = 0
                    dest_last_30d = 0
                    dest_updated_30d = 0
                    dest_last_grn = pd.NaT
                    dest_wh_grn = pd.NaT
                
                dest_grn_age = 0
                if pd.notna(dest_last_grn):
                    dest_grn_age = (pd.Timestamp.today() - pd.Timestamp(dest_last_grn)).days
                
                missing_recs.append({
                    'ITEM_CODE': item_code,
                    'Item Name': item_name,
                    'Group': item_group,
                    'Sub Group': item_subgroup,
                    'Source Shop': 'N/A',
                    'Source Stock': 0,
                    'Source Last 30d Sales': 0,
                    'Source Slow Moving': 'N/A',
                    'Source Fast Moving': 'N/A',
                    'Source Last GRN Date': 'N/A',
                    'Source Last WH GRN Date': 'N/A',
                    'Source GRN Age (days)': 0,
                    'Destination Shop': dest_shop,
                    'Destination Stock': dest_stock,
                    'Destination Last 30d Sales': dest_last_30d,
                    'Destination Last WH GRN Date': base_df[base_df['ITEM_CODE'] == item_code]['WH_GRN_DATE'].iloc[0] if not base_df[base_df['ITEM_CODE'] == item_code].empty and pd.notna(base_df[base_df['ITEM_CODE'] == item_code]['WH_GRN_DATE'].iloc[0]) else 'N/A',
                    'WH GRN +30 Date': (pd.Timestamp(base_df[base_df['ITEM_CODE'] == item_code]['WH_GRN_DATE'].iloc[0]) + pd.Timedelta(days=30)).strftime('%Y-%m-%d') if not base_df[base_df['ITEM_CODE'] == item_code].empty and pd.notna(base_df[base_df['ITEM_CODE'] == item_code]['WH_GRN_DATE'].iloc[0]) else 'N/A',
                    'Destination Sales (WH GRN +30d)': dest_updated_30d,
                    'Destination Sales Used': max(dest_last_30d, dest_updated_30d),
                    'Sales Source': 'Last 30d' if dest_last_30d >= dest_updated_30d else 'Updated 30d',
                    'Destination Last GRN Date': dest_last_grn if pd.notna(dest_last_grn) else 'N/A',
                    'Destination GRN Age (days)': dest_grn_age,
                    'Recommended_Qty': 0,
                    'Destination Updated Stock': dest_stock,
                    'Destination Final Stock In Hand Days': 0,
                    'Remark': 'ℹ️ No source available'
                })
        
        if missing_recs:
            result = pd.concat([result, pd.DataFrame(missing_recs)], ignore_index=True)
    
    # Sort by priority shop order
    priority_order = {shop: idx for idx, shop in enumerate(Config.PRIORITY_SHOPS)}
    result['_priority'] = result['Destination Shop'].map(priority_order).fillna(999)
    result = result.sort_values(['ITEM_CODE', '_priority', 'Source GRN Age (days)'], 
                                 ascending=[True, True, False], ignore_index=True)
    result = result.drop(columns=['_priority'])

    elapsed = time.time() - start_time
    logger.info(f"⚡ Generated {len(result)} recommendations in {elapsed:.2f}s")
    logger.info(f"📊 Source stock limits enforced for {len(source_allocations)} source combinations")
    
    return result


@st.cache_data(ttl=600, show_spinner=False)
def cached_generate_recommendations(group: str, subgroup: str, product: str, shop: str, item_type: str, supplier: str, item_name: str, use_grn_logic: bool, threshold: int = 10, use_cache: bool = True, limit: int = None, use_parallel: bool = True) -> pd.DataFrame:
    """
    Cacheable wrapper for ultra-fast recommendations with optional cache support.
    
    Performance:
    - With cache (single shop): 50-100ms (100x faster)
    - With indexes: 300-500ms (10x faster)  
    - Legacy fallback: 90-120s for 121K rows
    
    Args:
        use_cache: Whether to use pre-computed shop cache (default: True)
        limit: Maximum number of rows to return (for lazy loading)
        use_parallel: Whether to use multi-threaded execution (default: True)
    """
    import time
    import hashlib
    t0 = time.time()
    
    # Check in-memory cache first (MD5 hash of parameters)
    cache_key = hashlib.md5(
        f"{group}|{subgroup}|{product}|{shop}|{item_type}|{supplier}|{item_name}|{use_grn_logic}|{threshold}|{limit}".encode()
    ).hexdigest()
    
    if cache_key in st.session_state.rec_cache:
        cached_result, cache_time = st.session_state.rec_cache[cache_key]
        # Use cache if less than 5 minutes old
        if (time.time() - cache_time) < 300:
            logger.info(f"💾 MEMORY CACHE HIT: {len(cached_result)} recommendations (instant)")
            return cached_result.copy()
        else:
            # Expired - remove from cache
            del st.session_state.rec_cache[cache_key]
            logger.info("🔄 Cache expired, regenerating...")
    
    # Try optimized version first (uses inventory_master or cache)
    try:
        logger.info(f"🚀 Attempting ULTRA-FAST recommendation engine (cache={use_cache}, shop={shop}, limit={limit})...")
        result = generate_recommendations_optimized(
            group=group,
            subgroup=subgroup,
            product=product,
            shop=shop,
            use_grn_logic=use_grn_logic,
            threshold=threshold,
            use_cache=use_cache,
            limit=limit
        )
        
        if not result.empty:
            # Apply SIT filters if specified
            if item_type != 'All' or supplier != 'All' or item_name != 'All':
                logger.info("Applying SIT filters to optimized results...")
                sit_df = load_sit_filter_options()
                if not sit_df.empty:
                    # Filter by item codes that match SIT criteria
                    matched_codes = set()
                    
                    for _, row in sit_df.iterrows():
                        type_match = (item_type == 'All' or row['type'] == item_type)
                        supplier_match = (supplier == 'All' or row['vc_supplier_name'] == supplier)
                        name_match = (item_name == 'All' or row['item_name'] == item_name)
                        
                        if type_match and supplier_match and name_match:
                            matched_codes.add(row['item_code'].strip().upper())
                    
                    if matched_codes:
                        result = result[result['ITEM_CODE'].isin(matched_codes)]
            
            total_time = time.time() - t0
            logger.info(f"⚡ OPTIMIZED SUCCESS: {len(result)} recommendations in {total_time:.2f}s")
            
            # Store in memory cache for instant re-use
            st.session_state.rec_cache[cache_key] = (result.copy(), time.time())
            if len(st.session_state.rec_cache) > 10:  # Keep only 10 most recent
                oldest_key = min(st.session_state.rec_cache.keys(), key=lambda k: st.session_state.rec_cache[k][1])
                del st.session_state.rec_cache[oldest_key]
            
            return result
            
    except Exception as e:
        logger.warning(f"⚠️ Optimized engine unavailable: {e}")
        logger.info("Falling back to legacy recommendation engine...")
    
    # Legacy fallback - load dataframe and process with nested loops
    logger.info("🐌 Using LEGACY recommendation engine (slower)...")
    
    # Load the full inventory for the selected product/group/subgroup (all shops)
    t1 = time.time()
    full_df = load_inventory_data(group, subgroup, product, 'All')
    load_time = time.time() - t1
    logger.info(f"⏱️ load_inventory_data: {load_time:.3f}s ({len(full_df)} rows)")

    # Apply SIT filters if present
    t2 = time.time()
    sit_df = load_sit_filter_options()
    full_df = apply_itemdetails_filters(full_df, sit_df, item_type, supplier, item_name)
    sit_time = time.time() - t2
    logger.info(f"⏱️ apply_itemdetails_filters: {sit_time:.3f}s ({len(full_df)} rows)")
    
    if full_df.empty:
        logger.warning("⚠️ Filtered dataframe empty after SIT filters")
        return pd.DataFrame()

    # Call the non-cached generator (its result is cached by this wrapper)
    t3 = time.time()
    result = generate_recommendations(full_df, use_grn_logic, threshold)
    gen_time = time.time() - t3
    logger.info(f"⏱️ generate_recommendations: {gen_time:.3f}s ({len(result)} recommendations)")
    
    total_time = time.time() - t0
    logger.info(f"⏱️ TOTAL cached_generate_recommendations: {total_time:.3f}s")
    
    return result



# ============================================================
# EXPORT UTILITIES
# ============================================================

def convert_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def convert_recommendations_to_xlsx(df: pd.DataFrame) -> bytes:
    """Convert recommendations DataFrame to XLSX with conditional formatting for Source Expiry Date"""
    output = io.BytesIO()
    
    # Clean timezone info
    df_export = df.copy()
    for col in df_export.select_dtypes(include=['datetimetz']).columns:
        df_export[col] = df_export[col].dt.tz_localize(None)
    for col in df_export.select_dtypes(include=['object']).columns:
        if len(df_export[col].dropna()) > 0:
            first_val = df_export[col].dropna().iloc[0]
            if isinstance(first_val, pd.Timestamp) and hasattr(first_val, 'tz') and first_val.tz is not None:
                df_export[col] = pd.to_datetime(df_export[col]).dt.tz_localize(None)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Recommendations')
        
        # Apply conditional formatting if Source Expiry Date column exists
        if 'Source Expiry Date' in df_export.columns and 'Source Last GRN Date' in df_export.columns:
            workbook = writer.book
            worksheet = writer.sheets['Recommendations']
            
            # Define formats
            red_format = workbook.add_format({'bg_color': '#ffcccc'})
            yellow_format = workbook.add_format({'bg_color': '#ffffcc'})
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            
            # Get column indices
            expiry_col_idx = df_export.columns.get_loc('Source Expiry Date')
            grn_col_idx = df_export.columns.get_loc('Source Last GRN Date')
            expiry_days_col_idx = df_export.columns.get_loc('Source Expiry Days') if 'Source Expiry Days' in df_export.columns else None
            
            # Apply formatting row by row
            for row_idx in range(len(df_export)):
                excel_row = row_idx + 1  # Excel rows start at 1, +1 for header
                expiry_date = df_export.iloc[row_idx]['Source Expiry Date']
                expiry_days = df_export.iloc[row_idx]['Source Expiry Days'] if expiry_days_col_idx is not None else None
                
                # Check if should skip coloring
                skip_color = False
                if pd.isna(expiry_date) or expiry_date == '' or expiry_date == 0:
                    skip_color = True
                elif str(expiry_date) in ['1900-00-00 00:00:00', '1900-01-01 00:00:00', '1900-01-01', '1900-01-01 00:00:00.000000']:
                    skip_color = True
                
                if not skip_color and pd.notna(expiry_days):
                    try:
                        days = float(expiry_days)
                        
                        # Red if negative (expired)
                        if days < 0:
                            worksheet.write(excel_row, expiry_col_idx, expiry_date, red_format)
                        # Yellow if within 30 days (0-30)
                        elif 0 <= days <= 30:
                            worksheet.write(excel_row, expiry_col_idx, expiry_date, yellow_format)
                    except:
                        pass
    
    output.seek(0)
    return output.getvalue()

def convert_to_excel(slow_df, fast_df, final_df, filters_used):
    """Export the data to Excel with timezone-safe datetimes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # 🧹 Remove timezone info (Excel doesn’t support tz-aware datetimes)
        for df in [slow_df, fast_df, final_df]:
            for col in df.select_dtypes(include=['datetimetz']).columns:
                df[col] = df[col].dt.tz_localize(None)
            # Check object columns for timezone-aware timestamps  
            for col in df.select_dtypes(include=['object']).columns:
                if len(df[col].dropna()) > 0:
                    first_val = df[col].dropna().iloc[0]
                    if isinstance(first_val, pd.Timestamp) and hasattr(first_val, 'tz') and first_val.tz is not None:
                        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

        # Write sheets with row limits (Excel max: ~1M rows)
        SAFE_LIMIT = 1048000
        slow_df.head(SAFE_LIMIT).to_excel(writer, index=False, sheet_name='Slow_Moving_Shops')
        fast_df.head(SAFE_LIMIT).to_excel(writer, index=False, sheet_name='Fast_Moving_Shops')
        
        # Write Final_Recs sheet
        final_df_export = final_df.head(SAFE_LIMIT)
        final_df_export.to_excel(writer, index=False, sheet_name='Final_Recs')
        
        # Apply conditional formatting to Source Expiry Date column in Final_Recs
        if 'Source Expiry Date' in final_df.columns and 'Source Expiry Days' in final_df.columns:
            workbook = writer.book
            worksheet = writer.sheets['Final_Recs']
            
            # Define formats
            red_format = workbook.add_format({'bg_color': '#ffcccc'})
            yellow_format = workbook.add_format({'bg_color': '#ffffcc'})
            
            # Get column indices
            expiry_col_idx = final_df.columns.get_loc('Source Expiry Date')
            expiry_days_col_idx = final_df.columns.get_loc('Source Expiry Days')
            
            # Apply formatting row by row
            for row_idx, row in enumerate(final_df_export.itertuples(), start=1):  # Start at 1 (after header)
                expiry_date = getattr(row, final_df.columns[expiry_col_idx].replace(' ', '_'), None)
                expiry_days = getattr(row, final_df.columns[expiry_days_col_idx].replace(' ', '_'), None)
                
                # Check if should skip coloring
                skip_color = False
                if pd.isna(expiry_date) or expiry_date == '' or expiry_date == 0:
                    skip_color = True
                elif str(expiry_date) in ['1900-00-00 00:00:00', '1900-01-01 00:00:00', '1900-01-01', '1900-01-01 00:00:00.000000']:
                    skip_color = True
                
                if not skip_color and pd.notna(expiry_days):
                    try:
                        days = float(expiry_days)
                        
                        # Red if negative (expired)
                        if days < 0:
                            worksheet.write(row_idx, expiry_col_idx, expiry_date, red_format)
                        # Yellow if within 30 days (0-30)
                        elif 0 <= days <= 30:
                            worksheet.write(row_idx, expiry_col_idx, expiry_date, yellow_format)
                    except:
                        pass

        # Add filters and row count warnings
        meta_data = list(filters_used.items())
        meta_data.append(('---', '---'))
        meta_data.append(('Slow Moving Rows', f"{len(slow_df):,}"))
        meta_data.append(('Fast Moving Rows', f"{len(fast_df):,}"))
        meta_data.append(('Recommendations Rows', f"{len(final_df):,}"))
        
        if len(slow_df) > SAFE_LIMIT or len(fast_df) > SAFE_LIMIT or len(final_df) > SAFE_LIMIT:
            meta_data.append(('---', '---'))
            meta_data.append(('⚠️ WARNING', 'Some sheets truncated due to Excel row limit (~1M rows)'))
            if len(final_df) > SAFE_LIMIT:
                meta_data.append(('', f'Recommendations: {len(final_df):,} rows truncated to {SAFE_LIMIT:,}'))
            meta_data.append(('', 'Download CSV for complete data'))
        
        meta_df = pd.DataFrame(meta_data, columns=["Filter", "Value"])
        meta_df.to_excel(writer, index=False, sheet_name='Info')

    output.seek(0)
    return output


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    """Render header with logo centered and data freshness indicator - responsive"""
    # Get data freshness
    freshness = get_data_freshness()
    
    st.markdown(f"""
        <style>
        /* Main app background */
        .stApp {{
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #764ba2;
        }}
        
        /* Metric cards enhancement */
        [data-testid="stMetricValue"] {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        
        /* Selectbox/Dropdown styling - Fix visibility and size */
        .stSelectbox label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .stSelectbox > div > div {{
            background-color: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 15px;
        }}
        
        .stSelectbox [data-baseweb="select"] {{
            background-color: white;
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: white;
            border-color: #667eea;
            font-size: 15px;
            color: #333;
        }}
        
        /* Number input styling */
        .stNumberInput label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .stNumberInput input {{
            font-size: 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
        }}
        
        /* Checkbox styling */
        .stCheckbox label {{
            font-size: 15px;
            font-weight: 500;
            color: #333;
        }}
        
        /* Text input styling */
        .stTextInput label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .stTextInput input {{
            font-size: 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
        }}
        
        /* Caption text - make larger */
        .stCaptionContainer {{
            font-size: 14px !important;
        }}
        
        /* Improve main content width */
        .main .block-container {{
            max-width: 1400px;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        
        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        /* Pagination button styling - circular with gradient */
        button[key="prev_btn"], button[key="next_btn"] {{
            width: 45px !important;
            height: 45px !important;
            border-radius: 50% !important;
            padding: 0 !important;
            min-height: 45px !important;
            font-size: 20px !important;
        }}
        
        button[key="prev_btn"]:not(:disabled), button[key="next_btn"]:not(:disabled) {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }}
        
        button[key="prev_btn"]:not(:disabled):hover, button[key="next_btn"]:not(:disabled):hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }}
        
        button[key="prev_btn"]:disabled, button[key="next_btn"]:disabled {{
            background: transparent !important;
            color: #ccc !important;
            border: 2px solid #e0e0e0 !important;
            box-shadow: none !important;
        }}
        
        /* Download button */
        .stDownloadButton > button {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }}
        
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
        }}
        
        /* Clear Filter button styling */
        button[key="clear_chart_filter"] {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 6px 14px !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            margin-top: 8px !important;
            box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3) !important;
            transition: all 0.3s ease !important;
        }}
        
        button[key="clear_chart_filter"]:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4) !important;
        }}
        
        .melcom-logo {{
            width: 60px;
            height: 60px;
            margin-right: 15px;
            filter: drop-shadow(0 0 10px rgba(255,255,255,0.8));
            animation: pulse 2s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            font-size: 28px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: sticky !important;
            top: 0 !important;
            z-index: 999 !important;
        }}
        
        .title-content {{
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 1;
            gap: 15px;
        }}
        
        @media (max-width: 768px) {{
            .melcom-logo {{
                width: 40px;
                height: 40px;
                margin-right: 8px;
            }}
            .title-container {{
                font-size: 18px;
                padding: 15px;
                margin-bottom: 15px;
            }}
        }}
        .data-freshness {{
            position: sticky;
            top: 0;
            right: 20px;
            padding: 8px 12px;
            font-size: 11px;
            color: white;
            line-height: 1.4;
            margin-left: auto;
            z-index: 999;
            background: rgba(102, 126, 234, 0.3);
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }}
        
        .data-freshness-title {{
            font-weight: bold;
            margin-bottom: 4px;
            font-size: 12px;
            opacity: 0.95;
        }}
        
        .data-item {{
            display: flex;
            justify-content: space-between;
            margin: 2px 0;
            gap: 8px;
        }}
        
        .data-label {{
            font-weight: 500;
            opacity: 0.9;
        }}
        
        .data-value {{
            font-weight: 600;
        }}
        
        @media (max-width: 767px) {{
            .data-freshness {{
                position: static !important;
                margin: 15px auto 15px !important;
                max-width: 90% !important;
                background: rgba(255, 255, 255, 0.95) !important;
                color: #333 !important;
                border-radius: 8px !important;
                padding: 10px 12px !important;
                top: auto !important;
                right: auto !important;
            }}
            .data-freshness-title {{
                color: #667eea !important;
                font-size: 0.9rem !important;
            }}
            .data-item {{
                font-size: 0.8rem !important;
            }}
        }}
        </style>
        <div class="title-container">
            <div class="title-content">
                <img src="{Config.LOGO_URL}" alt="Logo" class="melcom-logo" />
                <span>MELCOM Inventory Pulse NO_WH</span>
            </div>
            <div class="data-freshness">
                <div class="data-freshness-title">📅 Data Updated Till</div>
                <div class="data-item">
                    <span class="data-label">🚚 GRN:</span>
                    <span class="data-value">{freshness['grn']}</span>
                </div>
                <div class="data-item">
                    <span class="data-label">💰 Sales:</span>
                    <span class="data-value">{freshness['sales']}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def change_password_in_db(employee_id: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Update password in users database after validation"""
    try:
        with get_db_connection('users') as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # First verify old password
                cursor.execute("""
                    SELECT employee_id FROM users
                    WHERE employee_id = %s AND password = %s AND LOWER(is_active) = 'true'
                """, (employee_id, old_password))
                
                if not cursor.fetchone():
                    return False, "Invalid employee ID or current password"
                
                # Update to new password
                cursor.execute("""
                    UPDATE users
                    SET password = %s
                    WHERE employee_id = %s
                """, (new_password, employee_id))
                
                conn.commit()
                
                # Clear authentication cache to force re-login with new password
                authenticate_user.clear()
                
                logger.info(f"✅ Password changed successfully for: {employee_id}")
                return True, "Password changed successfully! Please login with your new password."
                
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return False, f"Error updating password: {str(e)}"


def render_login():
    """Render login page: Employee ID above Password in a centered narrow column.

    The Login button is left-aligned under the password field.
    Press Enter to login functionality enabled.
    """
    
    # Initialize session state for change password mode
    if 'show_change_password' not in st.session_state:
        st.session_state.show_change_password = False
    
    # Custom styling for login page
    st.markdown("""
        <style>
        .login-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .login-header h1 {
            color: white;
            font-size: 32px;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .login-header p {
            color: rgba(255,255,255,0.9);
            font-size: 14px;
            margin-top: 10px;
        }
        </style>
        <div class="login-header">
            <h1>🔐 Melcom Inventory Pulse</h1>
            <p>Secure Access Portal</p>
        </div>
    """, unsafe_allow_html=True)

    # Create a centered narrow column to reduce input width
    left_col, center_col, right_col = st.columns([1, 0.6, 1])
    with center_col:
        # Toggle between login and change password
        if not st.session_state.show_change_password:
            # === LOGIN MODE ===
            with st.form(key="login_form", clear_on_submit=False):
                employee_id = st.text_input("👤 Employee ID")
                password = st.text_input("🔑 Password", type="password")

                # Left-aligned login button beneath the inputs
                btn_col1, btn_col2 = st.columns([1, 3])
                with btn_col1:
                    login_clicked = st.form_submit_button("Login", type="primary")
                with btn_col2:
                    st.write("")
                
                # Handle login when button clicked or Enter pressed
                if login_clicked:
                    if not employee_id or not password:
                        st.warning("⚠️ Enter both Employee ID and Password")
                    else:
                        with st.spinner("Authenticating..."):
                            user = authenticate_user(employee_id, password)

                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.success(f"✅ Welcome, {user['full_name']}")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
            
            # Change password link
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔑 Change Password", use_container_width=True):
                st.session_state.show_change_password = True
                st.rerun()
                
        else:
            # === CHANGE PASSWORD MODE ===
            st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #667eea;">🔑 Change Password</h3>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form(key="change_password_form", clear_on_submit=False):
                emp_id = st.text_input("👤 Employee ID")
                old_pwd = st.text_input("🔑 Current Password", type="password")
                new_pwd = st.text_input("🆕 New Password", type="password")
                confirm_pwd = st.text_input("✅ Confirm New Password", type="password")
                
                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    change_clicked = st.form_submit_button("Change Password", type="primary")
                with btn_col2:
                    cancel_clicked = st.form_submit_button("Cancel")
                
                if cancel_clicked:
                    st.session_state.show_change_password = False
                    st.rerun()
                
                if change_clicked:
                    # Validation
                    if not emp_id or not old_pwd or not new_pwd or not confirm_pwd:
                        st.warning("⚠️ Please fill in all fields")
                    elif new_pwd != confirm_pwd:
                        st.error("❌ New passwords do not match")
                    elif len(new_pwd) < 4:
                        st.warning("⚠️ Password must be at least 4 characters")
                    elif old_pwd == new_pwd:
                        st.warning("⚠️ New password must be different from current password")
                    else:
                        with st.spinner("Updating password..."):
                            success, message = change_password_in_db(emp_id, old_pwd, new_pwd)
                        
                        if success:
                            st.success(message)
                            st.session_state.show_change_password = False
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            # Back to login link
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.show_change_password = False
                st.rerun()

def render_base_filters(filter_df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Render base filters in the sidebar (Groups, Sub Group, Product Code, Shop Code)"""
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px; text-align: center; color: white; margin-bottom: 15px;">
            <div style="font-size: 16px; font-weight: bold;">🔍 Inventory Filters</div>
        </div>
    """, unsafe_allow_html=True)
    
    def add_all(options):
        return ['All'] + sorted(list(set([str(x).strip() for x in options if x and str(x).strip()])))
    
    # Check if a group was clicked from the chart
    group_list = add_all(filter_df['GROUPS'].dropna().unique())
    if st.session_state.get('chart_clicked_group'):
        clicked_group = st.session_state.chart_clicked_group
        # Set the group to clicked group if it exists in options
        if clicked_group in group_list:
            default_index = group_list.index(clicked_group)
            group = st.sidebar.selectbox("Groups", group_list, index=default_index, key="group_filter_clicked")
        else:
            group = st.sidebar.selectbox("Groups", group_list)
    else:
        group = st.sidebar.selectbox("Groups", group_list)
    
    subgroup_opts = filter_df['SUB_GROUP'].dropna().unique() if group == 'All' else \
                    filter_df.loc[filter_df['GROUPS'] == group, 'SUB_GROUP'].dropna().unique()
    subgroup = st.sidebar.selectbox("Sub Group", add_all(subgroup_opts))
    
    if subgroup == 'All' and group == 'All':
        product_opts = filter_df['ITEM_CODE'].dropna().unique()
    else:
        mask = ((filter_df['GROUPS'] == group) | (group == 'All')) & \
               ((filter_df['SUB_GROUP'] == subgroup) | (subgroup == 'All'))
        product_opts = filter_df.loc[mask, 'ITEM_CODE'].dropna().unique()
    product = st.sidebar.selectbox("Product Code", add_all(product_opts))
    
    if group == 'All' and subgroup == 'All' and product == 'All':
        shop_opts = filter_df['SHOP_CODE'].dropna().unique()
    else:
        mask = ((filter_df['GROUPS'] == group) | (group == 'All')) & \
               ((filter_df['SUB_GROUP'] == subgroup) | (subgroup == 'All')) & \
               ((filter_df['ITEM_CODE'] == product) | (product == 'All'))
        shop_opts = filter_df.loc[mask, 'SHOP_CODE'].dropna().unique()
    
    # Check if a shop was clicked from the chart
    if st.session_state.get('chart_clicked_shop'):
        clicked_shop = st.session_state.chart_clicked_shop
        # Set the shop code to clicked shop if it exists in options
        shop_list = add_all(shop_opts)
        if clicked_shop in shop_list:
            default_index = shop_list.index(clicked_shop)
            shop = st.sidebar.selectbox("Shop Code", shop_list, index=default_index, key="shop_filter_clicked")
        else:
            shop = st.sidebar.selectbox("Shop Code", shop_list)
    else:
        shop = st.sidebar.selectbox("Shop Code", add_all(shop_opts))
    
    return group, subgroup, product, shop

def render_sit_filters(sit_filter_df: pd.DataFrame) -> Tuple[str, str, str]:
    """Render SIT item-details filters on the main page - responsive"""

    
    def add_all(options):
        return ['All'] + sorted(list(set([str(x).strip() for x in options if x and str(x).strip()])))
    
    # Use responsive columns - will stack on mobile
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        item_type_opts = sit_filter_df['type'].dropna().unique() if not sit_filter_df.empty else []
        item_type = st.selectbox("Type", add_all(item_type_opts), key="sit_type")
    
    with col2:
        supplier_opts = sit_filter_df['vc_supplier_name'].dropna().unique() if not sit_filter_df.empty else []
        supplier = st.selectbox("Supplier Name", add_all(supplier_opts), key="sit_supplier")
    
    with col3:
        item_name_opts = sit_filter_df['item_name'].dropna().unique() if not sit_filter_df.empty else []
        item_name = st.selectbox("Item Name", add_all(item_name_opts), key="sit_item_name")
    
    return item_type, supplier, item_name

def render_filter_summary(group: str, subgroup: str, product: str, shop: str, item_type: str, supplier: str, item_name: str):
    """Render a heading showing currently applied filters with clear button"""
    active_filters = []
    has_chart_filter = False
    chart_filter_details = []
    
    if group and group != 'All':
        active_filters.append(f"Group: **{group}**")
        if st.session_state.get('chart_clicked_group'):
            has_chart_filter = True
            chart_filter_details.append(f"Group: **{group}**")
    if subgroup and subgroup != 'All':
        active_filters.append(f"Sub-Group: **{subgroup}**")
    if product and product != 'All':
        active_filters.append(f"Product: **{product}**")
    if shop and shop != 'All':
        active_filters.append(f"Shop: **{shop}**")
        if st.session_state.get('chart_clicked_shop'):
            has_chart_filter = True
            chart_filter_details.append(f"Shop: **{shop}**")
    if item_type and item_type != 'All':
        active_filters.append(f"Type: **{item_type}**")
    if supplier and supplier != 'All':
        active_filters.append(f"Supplier: **{supplier}**")
    if item_name and item_name != 'All':
        active_filters.append(f"Item: **{item_name}**")
    
    if active_filters:
        filter_text = " | ".join(active_filters)
        st.markdown(f"### 🎯 Active Filters: {filter_text}")
        
        # Show clear filter button if filters are applied via chart click
        if has_chart_filter:
            # Show which chart filters are active
            chart_filter_text = " + ".join(chart_filter_details)
            button_label = f"🧹 Clear Chart Filters ({chart_filter_text})" if len(chart_filter_details) > 1 else "🧹 Clear Filter for Melcom View"
            
            if st.button(button_label, key="clear_chart_filter", help="Reset chart-applied filters to show all data"):
                # Clear chart-triggered filters
                st.session_state.chart_clicked_shop = None
                st.session_state.chart_clicked_group = None
                st.session_state.trigger_generation = False
                st.session_state.auto_generate = False
                
                # Clear generated recommendations and related data
                if 'recommendations' in st.session_state:
                    del st.session_state.recommendations
                if 'final_recs' in st.session_state:
                    del st.session_state.final_recs
                if 'slow_shops' in st.session_state:
                    del st.session_state.slow_shops
                if 'fast_shops' in st.session_state:
                    del st.session_state.fast_shops
                if 'csv_data' in st.session_state:
                    del st.session_state.csv_data
                if 'excel_data' in st.session_state:
                    del st.session_state.excel_data
                if 'page_number' in st.session_state:
                    st.session_state.page_number = 0
                
                # Clear widget states to reset selectboxes to 'All'
                if 'shop_filter_clicked' in st.session_state:
                    del st.session_state.shop_filter_clicked
                if 'group_filter_clicked' in st.session_state:
                    del st.session_state.group_filter_clicked
                
                st.rerun()
    else:
        st.markdown("### 🎯 Active Filters: None (Showing All Data)")

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application"""
    
    # Page config
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide"
    )
    
    # Add comprehensive responsive CSS for all screen sizes
    st.markdown("""
        <style>
        /* ========================================
           ZOOM AND SCALE RESPONSIVE
           Makes dashboard adapt to browser zoom (Ctrl+/-)
        ======================================== */
        html {
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
            text-size-adjust: 100%;
        }
        
        body {
            font-size: clamp(12px, 1vw, 16px);
            overflow-x: hidden;
        }
        
        /* ========================================
           STICKY HEADER
        ======================================== */
        .title-container {
            position: sticky !important;
            top: 0 !important;
            z-index: 999 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Data freshness - position below sticky header */
        .data-freshness {
            z-index: 998 !important;
        }
        
        /* Main container - prevent overflow on zoom */
        .main .block-container {
            max-width: 100% !important;
            padding: clamp(1rem, 3vw, 2rem) clamp(0.5rem, 2vw, 1rem) !important;
        }
        
        /* Sidebar - scale with zoom */
        section[data-testid="stSidebar"] {
            width: clamp(200px, 20vw, 300px) !important;
        }
        
        section[data-testid="stSidebar"] .css-ng1t4o {
            width: clamp(200px, 20vw, 300px) !important;
        }
        
        /* Mobile: Collapse sidebar by default on phones */
        @media (max-width: 767px) {
            /* Hide sidebar initially on mobile */
            section[data-testid="stSidebar"] {
                margin-left: -100% !important;
                transition: margin-left 0.3s ease-in-out;
            }
            
            /* When sidebar is opened */
            section[data-testid="stSidebar"][aria-expanded="true"] {
                margin-left: 0 !important;
                width: 80% !important;
                max-width: 300px !important;
                z-index: 1000 !important;
                box-shadow: 2px 0 10px rgba(0,0,0,0.3);
            }
            
            /* Hamburger menu button styling */
            button[kind="header"] {
                display: block !important;
                position: fixed !important;
                top: 10px !important;
                left: 10px !important;
                z-index: 1001 !important;
                background: white !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
            }
            
            /* Make sidebar close button visible and prominent */
            section[data-testid="stSidebar"] button[kind="header"] {
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
                background: #667eea !important;
                color: white !important;
                border-radius: 50% !important;
                width: 40px !important;
                height: 40px !important;
                padding: 8px !important;
                position: absolute !important;
                top: 10px !important;
                right: 10px !important;
                z-index: 1002 !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
                border: 2px solid white !important;
            }
            
            /* Style the collapse icon */
            section[data-testid="stSidebar"] button[kind="header"] svg {
                fill: white !important;
                width: 20px !important;
                height: 20px !important;
            }
            
            /* Add visual indicator on hover/tap */
            section[data-testid="stSidebar"] button[kind="header"]:hover,
            section[data-testid="stSidebar"] button[kind="header"]:active {
                background: #764ba2 !important;
                transform: scale(1.1);
                transition: all 0.2s ease;
            }
            
            /* Main content - full width on mobile */
            .main .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                max-width: 100% !important;
            }
            
            /* Title container - adjust for mobile */
            .title-container {
                padding-left: 50px !important;
                font-size: 1rem !important;
            }
            
            .melcom-logo {
                height: 30px !important;
                width: 30px !important;
            }
        }
        
        /* Headings - fluid sizing */
        h1 {
            font-size: clamp(1.2rem, 3vw, 2.2rem) !important;
            line-height: 1.2 !important;
        }
        
        h2 {
            font-size: clamp(1rem, 2.5vw, 1.8rem) !important;
            line-height: 1.3 !important;
        }
        
        h3, h4 {
            font-size: clamp(0.9rem, 2vw, 1.5rem) !important;
            line-height: 1.4 !important;
        }
        
        /* Metrics - scale with zoom */
        [data-testid="stMetricValue"] {
            font-size: clamp(1.2rem, 2.5vw, 2.2rem) !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: clamp(0.7rem, 1.2vw, 1.1rem) !important;
        }
        
        /* Buttons - prevent overflow */
        .stButton > button,
        .stDownloadButton > button {
            font-size: clamp(0.8rem, 1vw, 1rem) !important;
            padding: clamp(0.4rem, 0.8vw, 0.6rem) clamp(0.8rem, 1.5vw, 1.2rem) !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Columns - stack on high zoom */
        div[data-testid="column"] {
            min-width: 0 !important;
            flex-shrink: 1;
        }
        
        /* Tables - scroll on zoom instead of break */
        .stDataFrame {
            width: 100% !important;
            overflow-x: auto !important;
            font-size: clamp(0.7rem, 1vw, 1rem) !important;
        }
        
        .dataframe-container {
            overflow-x: auto !important;
            max-width: 100vw !important;
        }
        
        /* Selectbox and inputs - scale properly */
        .stSelectbox label,
        .stNumberInput label,
        .stMultiSelect label {
            font-size: clamp(0.8rem, 1.1vw, 1rem) !important;
        }
        
        .stSelectbox select,
        .stNumberInput input {
            font-size: clamp(0.8rem, 1vw, 0.95rem) !important;
        }
        
        /* Tabs - prevent overflow */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: clamp(0.8rem, 1vw, 1rem) !important;
            padding: clamp(0.3rem, 0.6vw, 0.5rem) clamp(0.8rem, 1.5vw, 1rem) !important;
        }
        
        /* Charts - scale with container */
        .js-plotly-plot {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        .js-plotly-plot .plotly {
            width: 100% !important;
        }
        
        /* Caption text - scale */
        .caption, [class*="caption"] {
            font-size: clamp(0.7rem, 0.9vw, 0.85rem) !important;
        }
        
        /* Info boxes - scale */
        .stAlert, [data-testid="stAlert"] {
            font-size: clamp(0.75rem, 1vw, 0.9rem) !important;
            padding: clamp(0.5rem, 1vw, 0.75rem) !important;
        }
        
        /* Expander - scale */
        .streamlit-expanderHeader {
            font-size: clamp(0.85rem, 1.1vw, 1rem) !important;
        }
        
        /* Base styles - Desktop (1601px+) */
        
        /* Format dataframe columns properly */
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td,
        .stDataFrame div[data-testid="stDataFrameResizable"] table thead tr th {
            text-align: center !important;
        }
        
        /* Column headers: wrap text and center align vertically */
        .stDataFrame div[data-testid="stDataFrameResizable"] table thead tr th {
            white-space: normal !important;
            word-wrap: break-word !important;
            vertical-align: middle !important;
            text-align: center !important;
            padding: 8px 5px !important;
            line-height: 1.3 !important;
        }
        
        /* Left align text columns (Item Name, Shop codes, Remark) */
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td:nth-child(1),
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td:nth-child(2) {
            text-align: left !important;
            padding-left: 10px !important;
        }
        
        /* Right align numeric columns */
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td:has(div:matches('[0-9]+')) {
            text-align: right !important;
            padding-right: 10px !important;
        }
        
        /* Make tables horizontally scrollable on all devices */
        .dataframe-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        .stDataFrame {
            overflow-x: auto;
        }
        
        /* Plotly charts responsive */
        .js-plotly-plot {
            width: 100% !important;
        }
        
        /* ========================================
           SMALL PHONES (320px - 480px)
           iPhone SE, small Androids
        ======================================== */
        @media (max-width: 480px) {
            body {
                font-size: 12px;
            }
            
            h1 {
                font-size: 1.2rem !important;
            }
            
            h2 {
                font-size: 1.1rem !important;
            }
            
            h3, h4 {
                font-size: 1rem !important;
            }
            
            /* Stack all columns vertically */
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 1rem;
            }
            
            /* Full width buttons */
            .stButton > button,
            .stDownloadButton > button {
                width: 100% !important;
                margin: 5px 0;
                font-size: 0.9rem;
            }
            
            /* Compact selectboxes */
            .row-widget.stSelectbox {
                width: 100%;
            }
            
            /* Smaller metrics */
            [data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 0.8rem !important;
            }
            
            /* Hide pie chart on very small screens, keep table full width */
            .stDataFrame {
                font-size: 0.75rem;
            }
            
            /* Sidebar adjustments */
            section[data-testid="stSidebar"] {
                min-width: 100% !important;
            }
        }
        
        /* ========================================
           LARGE PHONES (481px - 767px)
           iPhone 17 Pro Max, Samsung Z Fold (unfolded inner), Pixel
        ======================================== */
        @media (min-width: 481px) and (max-width: 767px) {
            body {
                font-size: 14px;
            }
            
            h1 {
                font-size: 1.4rem !important;
            }
            
            h2 {
                font-size: 1.2rem !important;
            }
            
            h3 {
                font-size: 1.1rem !important;
            }
            
            /* Stack columns on large phones */
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 0.75rem;
            }
            
            .stButton > button,
            .stDownloadButton > button {
                width: 100%;
                margin: 5px 0;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 1.4rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 0.85rem !important;
            }
            
            .stDataFrame {
                font-size: 0.8rem;
            }
        }
        
        /* ========================================
           TABLETS - Portrait (768px - 1024px)
           iPad, iPad Air, Samsung Galaxy Tab
        ======================================== */
        @media (min-width: 768px) and (max-width: 1024px) {
            body {
                font-size: 15px;
            }
            
            h1 {
                font-size: 1.6rem !important;
            }
            
            h2 {
                font-size: 1.4rem !important;
            }
            
            h3 {
                font-size: 1.2rem !important;
            }
            
            /* Two-column layout for tablets */
            div[data-testid="column"] {
                padding: 0 0.5rem;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 1.6rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 0.9rem !important;
            }
            
            .stDataFrame {
                font-size: 0.85rem;
            }
            
            /* Adjust pie chart size */
            .js-plotly-plot {
                max-height: 350px;
            }
        }
        
        /* ========================================
           LARGE TABLETS / SMALL LAPTOPS (1025px - 1366px)
           iPad Pro, Surface Pro, small laptops
        ======================================== */
        @media (min-width: 1025px) and (max-width: 1366px) {
            body {
                font-size: 15px;
            }
            
            h1 {
                font-size: 1.8rem !important;
            }
            
            h2 {
                font-size: 1.5rem !important;
            }
            
            h3 {
                font-size: 1.3rem !important;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 1.8rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 0.95rem !important;
            }
            
            .stDataFrame {
                font-size: 0.9rem;
            }
            
            .js-plotly-plot {
                max-height: 400px;
            }
        }
        
        /* ========================================
           STANDARD LAPTOPS (1367px - 1600px)
           13-15 inch laptops, standard displays
        ======================================== */
        @media (min-width: 1367px) and (max-width: 1600px) {
            body {
                font-size: 16px;
            }
            
            h1 {
                font-size: 2rem !important;
            }
            
            h2 {
                font-size: 1.6rem !important;
            }
            
            h3 {
                font-size: 1.4rem !important;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 2rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 1rem !important;
            }
            
            .stDataFrame {
                font-size: 0.95rem;
            }
        }
        
        /* ========================================
           LARGE SCREENS (1601px+)
           Large monitors, 4K displays
        ======================================== */
        @media (min-width: 1601px) {
            body {
                font-size: 16px;
            }
            
            h1 {
                font-size: 2.2rem !important;
            }
            
            h2 {
                font-size: 1.8rem !important;
            }
            
            h3 {
                font-size: 1.5rem !important;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 2.2rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 1.1rem !important;
            }
            
            .stDataFrame {
                font-size: 1rem;
            }
        }
        
        /* ========================================
           LANDSCAPE ORIENTATION ADJUSTMENTS
        ======================================== */
        @media (max-width: 1024px) and (orientation: landscape) {
            /* Optimize for landscape mode on tablets/phones */
            div[data-testid="column"] {
                padding: 0 0.3rem;
            }
            
            h1, h2, h3 {
                margin-top: 0.5rem !important;
                margin-bottom: 0.5rem !important;
            }
        }
        
        /* ========================================
           TOUCH DEVICE OPTIMIZATIONS
        ======================================== */
        @media (hover: none) and (pointer: coarse) {
            /* Larger touch targets for mobile devices */
            .stButton > button,
            .stDownloadButton > button {
                min-height: 44px;
                padding: 12px 20px;
            }
            
            /* Better spacing for touch */
            .stSelectbox, .stNumberInput, .stCheckbox {
                margin-bottom: 1rem;
            }
        }
        
        /* ========================================
           PRINT STYLES
        ======================================== */
        @media print {
            /* Hide sidebar and buttons when printing */
            section[data-testid="stSidebar"],
            .stButton,
            .stDownloadButton {
                display: none !important;
            }
            
            /* Full width for content */
            .main .block-container {
                max-width: 100% !important;
                padding: 0 !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    if "chart_clicked_shop" not in st.session_state:
        st.session_state.chart_clicked_shop = None
    if "chart_clicked_group" not in st.session_state:
        st.session_state.chart_clicked_group = None
    if "trigger_generation" not in st.session_state:
        st.session_state.trigger_generation = False
    if "previous_filters" not in st.session_state:
        st.session_state.previous_filters = {}
    if "auto_generate" not in st.session_state:
        st.session_state.auto_generate = False
    
    # Login flow
    if not st.session_state.logged_in:
        render_login()
        st.stop()
    
    # Access control
    user = st.session_state.user
    if not check_table_access(user, "nowhstock_tbl_new"):
        st.error("🚫 Access denied")
        st.stop()
    
    # Header
    render_header()
    
    # Sidebar
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; color: white;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">👋 Welcome</div>
            <div style="font-size: 14px;">{}</div>
            <div style="font-size: 12px; opacity: 0.9; margin-top: 3px;">ID: {}</div>
        </div>
    """.format(user['full_name'], user['employee_id']), unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout", key="logout_btn", width='stretch'):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()
    st.sidebar.divider()
    
    # Load filters
    filter_df = load_filter_options()
    sit_filter_df = load_sit_filter_options()
    
    if filter_df.empty:
        st.error("❌ Cannot load filters")
        st.stop()
    
    # Render base filters in sidebar
    group, subgroup, product, shop = render_base_filters(filter_df)
    
    # Detect if filters changed from chart clicks
    current_filters = {'group': group, 'shop': shop}
    filters_changed_from_chart = False
    
    if st.session_state.get('chart_clicked_shop') or st.session_state.get('chart_clicked_group'):
        if st.session_state.previous_filters != current_filters:
            filters_changed_from_chart = True
            st.session_state.previous_filters = current_filters
            st.session_state.auto_generate = True
    
    # Additional controls in sidebar
    threshold = st.sidebar.number_input("Sales threshold (Fast vs Slow)", min_value=0, value=Config.DEFAULT_THRESHOLD)
    use_grn_logic = st.sidebar.checkbox("📦 Use GRN Date Logic", value=True)
    show_grn_info = st.sidebar.checkbox("🔍 Show GRN Info", value=True)
    
    # Calculate dates for display
    end_date = datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    st.sidebar.caption(f"✨ Sales period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Render SIT filters on main page (BEFORE loading inventory to avoid delays)
    st.markdown("---")

    
    def add_all(options):
        return ['All'] + sorted(list(set([str(x).strip() for x in options if x and str(x).strip()])))
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        item_type_opts = sit_filter_df['type'].dropna().unique() if not sit_filter_df.empty else []
        item_type = st.selectbox("Type", add_all(item_type_opts), key="sit_type")
    
    with col2:
        supplier_opts = sit_filter_df['vc_supplier_name'].dropna().unique() if not sit_filter_df.empty else []
        supplier = st.selectbox("Supplier Name", add_all(supplier_opts), key="sit_supplier")
    
    with col3:
        item_name_opts = sit_filter_df['item_name'].dropna().unique() if not sit_filter_df.empty else []
        item_name = st.selectbox("Product Name", add_all(item_name_opts), key="sit_item_name")
    
    # Load data
    inventory_df = load_inventory_data(group, subgroup, product, shop)

    # Apply Item Details (SIT) filters if selected
    inventory_df = apply_itemdetails_filters(inventory_df, sit_filter_df, item_type, supplier, item_name)
    
    # Show active filters heading
    render_filter_summary(group, subgroup, product, shop, item_type, supplier, item_name)
    
    st.divider()
    
    # Merge TYPE information from mv_item_meta for pie chart
    if not sit_filter_df.empty:
        # Create a mapping of ITEM_CODE to TYPE
        sit_type_map = sit_filter_df.copy()
        sit_type_map['ITEM_CODE'] = sit_type_map['item_code'].astype(str).str.strip().str.upper()
        sit_type_map = sit_type_map[['ITEM_CODE', 'type']].drop_duplicates(subset=['ITEM_CODE'])
        sit_type_map.columns = ['ITEM_CODE', 'TYPE']
        
        # Merge TYPE into inventory_df
        inventory_df = inventory_df.merge(sit_type_map, on='ITEM_CODE', how='left')
        inventory_df['TYPE'] = inventory_df['TYPE'].fillna('Unknown')

    if inventory_df.empty:
        st.warning("No records found")
        st.stop()
    
    # Classify
    inventory_df['Sales_Status'] = np.where(
        inventory_df['ITEM_SALES_30_DAYS'] >= threshold,
        'Fast',
        'Slow'
    )
    slow_shops = inventory_df[inventory_df['Sales_Status'] == 'Slow']
    fast_shops = inventory_df[inventory_df['Sales_Status'] == 'Fast']
    
    # Calculate trend metrics with unique shop and item counts
    slow_unique_shops = slow_shops['SHOP_CODE'].nunique() if not slow_shops.empty else 0
    fast_unique_shops = fast_shops['SHOP_CODE'].nunique() if not fast_shops.empty else 0
    slow_unique_items = slow_shops['ITEM_CODE'].nunique() if not slow_shops.empty else 0
    fast_unique_items = fast_shops['ITEM_CODE'].nunique() if not fast_shops.empty else 0
    slow_total_stock = int(slow_shops['SHOP_STOCK'].sum()) if not slow_shops.empty else 0
    fast_total_stock = int(fast_shops['SHOP_STOCK'].sum()) if not fast_shops.empty else 0
    total_unique_shops = inventory_df['SHOP_CODE'].nunique()
    slow_pct = (slow_unique_shops / total_unique_shops * 100) if total_unique_shops > 0 else 0
    fast_pct = (fast_unique_shops / total_unique_shops * 100) if total_unique_shops > 0 else 0
    
    # Display Current Inventory Trends and Shop-wise Analysis side by side
    col_trends, col_chart = st.columns([1, 4])
    
    with col_trends:
        # Add filter indicator if filters are active
        filter_indicator = ""
        if shop != "All" or group != "All":
            filter_parts = []
            if shop != "All":
                filter_parts.append(f"Shop={shop}")
            if group != "All":
                filter_parts.append(f"Group={group}")
            filter_text = " | ".join(filter_parts)
            filter_indicator = f'<div style="text-align: center; font-size: 11px; color: #667eea; margin-bottom: 8px;">📍 Filtered: {filter_text}</div>'
        
        # Use st.html for better rendering
        st.html(f"""
        <style>
        @media only screen and (max-width: 768px) {{
            [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
            .stPlotlyChart {{
                overflow-x: auto !important;
            }}
            div[style*="grid-template-columns"] {{
                grid-template-columns: 1fr !important;
            }}
        }}
        </style>
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: -10px 15px; border-radius: 12px; border-left: 5px solid #667eea; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.07);">
            <div style="text-align: center; font-size: 18px; font-weight: bold; color: #333; margin-bottom: 12px;">📊 Current Inventory Trends</div>
            {filter_indicator}
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                <div style="background: white; padding: 12px 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 3px solid #f5576c;">
                    <div style="font-size: 12px; color: #444; margin-bottom: 5px; font-weight: 600;">Slow Moving Shops</div>
                    <div style="font-size: 28px; font-weight: bold; color: #f5576c; margin-bottom: 3px;">{slow_unique_shops}</div>
                    <div style="font-size: 13px; color: #555; margin-bottom: 6px;">{slow_pct:.1f}% of all shops</div>
                    <div style="font-size: 12px; color: #666; padding-top: 5px; border-top: 1px solid #f0f0f0;">
                        <div><strong>{slow_unique_items}</strong> unique items</div>
                        <div style="margin-top: 2px;"><strong>{slow_total_stock:,}</strong> total stock units</div>
                    </div>
                </div>
                <div style="background: white; padding: 12px 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 3px solid #10b981;">
                    <div style="font-size: 12px; color: #444; margin-bottom: 5px; font-weight: 600;">Fast Moving Shops</div>
                    <div style="font-size: 28px; font-weight: bold; color: #10b981; margin-bottom: 3px;">{fast_unique_shops}</div>
                    <div style="font-size: 13px; color: #555; margin-bottom: 6px;">{fast_pct:.1f}% of all shops</div>
                    <div style="font-size: 12px; color: #666; padding-top: 5px; border-top: 1px solid #f0f0f0;">
                        <div><strong>{fast_unique_items}</strong> unique items</div>
                        <div style="margin-top: 2px;"><strong>{fast_total_stock:,}</strong> total stock units</div>
                    </div>
                </div>
                <div style="background: white; padding: 12px 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 3px solid #667eea;">
                    <div style="font-size: 12px; color: #444; margin-bottom: 5px; font-weight: 600;">Transfer Potential</div>
                    <div style="font-size: 28px; font-weight: bold; color: #667eea; margin-bottom: 3px;">{len(slow_shops)}</div>
                    <div style="font-size: 13px; color: #555; margin-bottom: 6px;">slow moving records</div>
                    <div style="font-size: 12px; color: #666; padding-top: 5px; border-top: 1px solid #f0f0f0;">
                        <div>Ready for reallocation</div>
                        <div style="margin-top: 2px;">to fast-moving shops</div>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    with col_chart:
        # Determine x-axis based on shop filter
        if shop != 'All':
            # When specific shop is selected, show groups on x-axis
            subtitle_text = f"Unique item codes by Group for Shop: {shop}"
            shop_filtered = inventory_df[inventory_df['SHOP_CODE'] == shop]
            
            slow_by_group = shop_filtered[shop_filtered['Sales_Status'] == 'Slow'].groupby('GROUPS')['ITEM_CODE'].nunique().reset_index()
            slow_by_group.columns = ['Group', 'Slow Moving Items']
            
            fast_by_group = shop_filtered[shop_filtered['Sales_Status'] == 'Fast'].groupby('GROUPS')['ITEM_CODE'].nunique().reset_index()
            fast_by_group.columns = ['Group', 'Fast Moving Items']
            
            analysis_df = slow_by_group.merge(fast_by_group, on='Group', how='outer').fillna(0)
            analysis_df = analysis_df.sort_values('Slow Moving Items', ascending=False)
            x_data = analysis_df['Group']
            x_title = 'Product Group'
        else:
            # When All shops, show shops on x-axis
            subtitle_text = "Unique item codes per shop: Slow Moving vs Fast Moving"
            slow_by_shop = slow_shops.groupby('SHOP_CODE')['ITEM_CODE'].nunique().reset_index()
            slow_by_shop.columns = ['Shop', 'Slow Moving Items']
            
            fast_by_shop = fast_shops.groupby('SHOP_CODE')['ITEM_CODE'].nunique().reset_index()
            fast_by_shop.columns = ['Shop', 'Fast Moving Items']
            
            analysis_df = slow_by_shop.merge(fast_by_shop, on='Shop', how='outer').fillna(0)
            analysis_df = analysis_df.sort_values('Slow Moving Items', ascending=False)
            x_data = analysis_df.get('Shop', analysis_df.get('Group', []))
            x_title = 'Shop Code'
        
        # Create dual-axis bar chart with secondary y-axis for fast moving
        fig = go.Figure()
        
        # Calculate max values for proper scaling
        max_slow = analysis_df['Slow Moving Items'].max()
        max_fast = analysis_df['Fast Moving Items'].max()
        
        # Primary y-axis: Slow Moving (left side)
        fig.add_trace(go.Bar(
            name='Slow Moving',
            x=x_data,
            y=analysis_df['Slow Moving Items'],
            marker_color='#ff6b6b',
            text=analysis_df['Slow Moving Items'].astype(int),
            textposition='inside',
            textfont=dict(color='white', size=10),
            yaxis='y',
            offsetgroup=1
        ))
        
        # Secondary y-axis: Fast Moving (right side)
        fig.add_trace(go.Bar(
            name='Fast Moving',
            x=x_data,
            y=analysis_df['Fast Moving Items'],
            marker_color='#51cf66',
            text=analysis_df['Fast Moving Items'].astype(int),
            textposition='inside',
            textfont=dict(color='white', size=10),
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.update_layout(
            title=dict(
                text='Shop Slow vs Fast Moving<br><sub>(unique item code)</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#333', family='Arial, sans-serif')
            ),
            xaxis_title=dict(
                text=f"{x_title}<br><sup style='font-size:11px;color:#666;'>{subtitle_text}</sup>",
                font=dict(size=13)
            ),
            yaxis=dict(
                title=dict(text='Slow Moving Items', font=dict(color='#ff6b6b')),
                tickfont=dict(color='#ff6b6b'),
                side='left',
                range=[0, max_slow * 1.15]
            ),
            yaxis2=dict(
                title=dict(text='Fast Moving Items', font=dict(color='#51cf66')),
                tickfont=dict(color='#51cf66'),
                overlaying='y',
                side='right',
                range=[0, max_fast * 1.15]
            ),
            barmode='group',
            height=280,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=50, b=40, l=50, r=50),
            bargap=0.3,
            bargroupgap=0.1,
            clickmode='event+select'
        )
        
        # Display the chart and capture click events
        chart_event = st.plotly_chart(fig, use_container_width=True, key="shop_chart", on_select="rerun")
        
        # Handle chart click events
        if chart_event and chart_event.selection and chart_event.selection.points:
            clicked_point = chart_event.selection.points[0]
            clicked_value = str(clicked_point.get('x', '')).strip().upper()
            
            if clicked_value:
                # Check if it's a shop code (when shop filter is 'All') or group (when specific shop is selected)
                if shop != 'All':
                    # Clicked on a group - apply group filter (preserve existing shop filter)
                    st.session_state.chart_clicked_group = clicked_value
                    # Keep the existing shop filter if it was set from chart
                    # (chart_clicked_shop is already in session state if previously set)
                    st.session_state.trigger_generation = True
                    st.rerun()
                else:
                    # Clicked on a shop code
                    if clicked_value in Config.PRIORITY_SHOPS:
                        st.warning(f"⚠️ **{clicked_value}** is a priority shop. Transferring FROM priority shops is not allowed.")
                    else:
                        # Store clicked shop and trigger recommendation generation (preserve existing group filter)
                        st.session_state.chart_clicked_shop = clicked_value
                        # Keep the existing group filter if it was set from chart
                        # (chart_clicked_group is already in session state if previously set)
                        st.session_state.trigger_generation = True
                        st.rerun()
    
    # Multi-page tabs for Slow Moving and Fast Moving with custom font size and active tab highlighting
    st.markdown("""
        <style>
        /* Increase tab font size */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 18px;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        /* Highlight active/selected tab */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
            color: white;
        }
        /* Inactive tab styling */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
            background-color: #f0f0f0;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"]:hover {
            background-color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([f"📉 Slow Moving Shops (Sales < {threshold})", f"📈 Fast Moving Shops (Sales >= {threshold})"])
    
    with tab1:
        st.caption(f"{len(slow_shops)} records | Total Sales: {int(slow_shops['ITEM_SALES_30_DAYS'].sum()):,} | Total Stock: {slow_total_stock:,}")
        
        # Create two columns: table on left, pie chart on right
        col_table, col_chart = st.columns([3, 1])
        
        with col_chart:
            # Create pie chart for Direct Import vs Local (slow-moving items only)
            if not slow_shops.empty and 'TYPE' in slow_shops.columns:
                # Get unique items count by TYPE
                type_counts = slow_shops.groupby('TYPE')['ITEM_CODE'].nunique().reset_index()
                type_counts.columns = ['Type', 'Unique Items']
                
                # Calculate percentages
                total_unique = type_counts['Unique Items'].sum()
                type_counts['Percentage'] = (type_counts['Unique Items'] / total_unique * 100).round(1)
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=type_counts['Type'],
                    values=type_counts['Unique Items'],
                    hole=0.4,
                    marker=dict(
                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
                        line=dict(color='white', width=2)
                    ),
                    textinfo='label+percent',
                    textposition='outside',
                    hovertemplate='<b>%{label}</b><br>' +
                                  'Items: %{value:,}<br>' +
                                  'Percentage: %{percent}<br>' +
                                  '<extra></extra>'
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Slow-Moving Items by Type<br><sub>{total_unique:,} Unique Items</sub>',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 14, 'color': '#333'}
                    },
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.05,
                        font=dict(size=10)
                    ),
                    margin=dict(l=10, r=10, t=80, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True, key="slow_type_pie")
            else:
                st.info("📊 Type information not available")
        
        with col_table:
            if not slow_shops.empty:
                # Very compact pixel widths to match screenshot
                col_config = {}
                for col in slow_shops.columns:
                    if col == 'SHOP_CODE':
                        col_config[col] = st.column_config.TextColumn(col, width=50)
                    elif col == 'ITEM_CODE':
                        col_config[col] = st.column_config.TextColumn(col, width=50)
                    elif col == 'ITEM_NAME':
                        col_config[col] = st.column_config.TextColumn(col, width=400)
                    elif col == 'SHOP_STOCK':
                        col_config[col] = st.column_config.NumberColumn(col, width=60)
                    elif col == 'ITEM_SALES_30_DAYS':
                        col_config[col] = st.column_config.NumberColumn(col, width=150)
                    elif col == 'Sales_Status':
                        col_config[col] = st.column_config.TextColumn(col, width=100)
                    elif col in ['GROUPS']:
                        col_config[col] = st.column_config.TextColumn(col, width=190)
                    elif col in ['SUB_GROUP']:    
                        col_config[col] = st.column_config.TextColumn(col, width=300)
                    elif col == 'SHOP_GRN_DATE':
                        col_config[col] = st.column_config.Column(col, width=170)
                    elif col == 'TYPE':
                        col_config[col] = st.column_config.TextColumn(col, width=120)
                    else:
                        col_config[col] = st.column_config.Column(col, width=60)
                
                st.dataframe(slow_shops, height=400, hide_index=True, column_config=col_config, use_container_width=True)
            else:
                st.info("✅ No slow-moving items found")
    
    with tab2:
        st.caption(f"{len(fast_shops)} records | Total Sales: {int(fast_shops['ITEM_SALES_30_DAYS'].sum()):,} | Total Stock: {fast_total_stock:,}")
        if not fast_shops.empty:
            # Very compact pixel widths to match screenshot
            col_config = {}
            for col in fast_shops.columns:
                if col == 'SHOP_CODE':
                    col_config[col] = st.column_config.TextColumn(col, width=50)
                elif col == 'ITEM_CODE':
                    col_config[col] = st.column_config.TextColumn(col, width=50)
                elif col == 'ITEM_NAME':
                    col_config[col] = st.column_config.TextColumn(col, width=400)
                elif col == 'SHOP_STOCK':
                    col_config[col] = st.column_config.NumberColumn(col, width=60)
                elif col == 'ITEM_SALES_30_DAYS':
                    col_config[col] = st.column_config.NumberColumn(col, width=150)
                elif col == 'Sales_Status':
                    col_config[col] = st.column_config.TextColumn(col, width=100)
                elif col in ['GROUPS']:
                    col_config[col] = st.column_config.TextColumn(col, width=190)
                elif col in ['SUB_GROUP']:    
                    col_config[col] = st.column_config.TextColumn(col, width=300)
                elif col == 'SHOP_GRN_DATE':
                    col_config[col] = st.column_config.Column(col, width=170)
                else:
                    col_config[col] = st.column_config.Column(col, width=60)
            
            st.dataframe(fast_shops, height=400, hide_index=True, column_config=col_config, use_container_width=True)
        else:
            st.info(f"ℹ️ No items with sales >= {threshold}. Try lowering the threshold.")
    
    st.divider()

    # Generate recommendations section
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Show prominent info when filters are applied from chart
    if st.session_state.get('chart_clicked_shop') or st.session_state.get('chart_clicked_group'):
        active_chart_filters = []
        if st.session_state.get('chart_clicked_shop'):
            active_chart_filters.append(f"Shop: **{st.session_state.chart_clicked_shop}**")
        if st.session_state.get('chart_clicked_group'):
            active_chart_filters.append(f"Group: **{st.session_state.chart_clicked_group}**")
        
        filter_info = " + ".join(active_chart_filters)
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">🎯 Chart Filters Applied</div>
                <div style="font-size: 14px;">{filter_info}</div>
                <div style="font-size: 12px; opacity: 0.9; margin-top: 5px;">Data above is filtered. Click button below for transfer recommendations.</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Checkbox to generate complete recommendations without any limit
    generate_all = st.checkbox(
        "📊 Generate ALL recommendations without limit",
        value=False,
        key="generate_all_checkbox",
        help="When checked, generates complete recommendations based on selected filters without any row limit. May take longer for large datasets."
    )
    
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 1.5])
    with btn_col2:
        gen_clicked = st.button("🎯 Generate Smart Recommendations", key="gen_rec_btn", width='stretch', type="primary")
    st.markdown("""
        <div style="text-align: center; margin-top: 8px; margin-bottom: 15px; font-size: 13px; color: #666;">
            Analyze inventory and suggest optimal transfers
        </div>
    """, unsafe_allow_html=True)

    # Auto-trigger generation if shop/group was clicked from chart or if auto_generate flag is set
    if st.session_state.get('trigger_generation'):
        gen_clicked = True
        st.session_state.trigger_generation = False  # Reset trigger
    
    if st.session_state.get('auto_generate'):
        gen_clicked = True
        st.session_state.auto_generate = False  # Reset auto-generate flag

    if gen_clicked:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📊 Loading data...")
            progress_bar.progress(25)

            if product != "All":
                full_df = load_inventory_data(group, subgroup, product, "All")
                # Apply SIT filters to the full set used for recommendations
                full_df = apply_itemdetails_filters(full_df, sit_filter_df, item_type, supplier, item_name)
            else:
                full_df = inventory_df

            status_text.text("⚡ Generating recommendations...")
            progress_bar.progress(60)

            # Determine if lazy loading needed based on checkbox
            if generate_all:
                # Checkbox CHECKED: Generate ALL recommendations without any limit
                limited_load = False
                row_limit = None
                status_text.text("⚡ Generating ALL recommendations (no limit)...")
            else:
                # Checkbox UNCHECKED: Use existing logic (50k limit when all filters = 'All')
                limited_load = (group == 'All' and subgroup == 'All' and product == 'All' and shop == 'All')
                row_limit = 50000 if limited_load else None
            
            # Use cached wrapper which tries cache first for single shop queries
            recommendations = cached_generate_recommendations(
                group=group,
                subgroup=subgroup,
                product=product,
                shop=shop,
                item_type=item_type,
                supplier=supplier,
                item_name=item_name,
                use_grn_logic=use_grn_logic,
                threshold=threshold,
                use_cache=True,  # Use cache for instant shop-specific results
                limit=row_limit,  # Use determined limit based on checkbox
                use_parallel=True  # Enable multi-threading
            )
            
            # CRITICAL: Filter recommendations to ONLY priority shops immediately after generation
            if not recommendations.empty and 'Destination Shop' in recommendations.columns:
                initial_count = len(recommendations)
                non_priority = recommendations[~recommendations['Destination Shop'].isin(Config.PRIORITY_SHOPS)]
                
                if len(non_priority) > 0:
                    non_priority_shops = non_priority['Destination Shop'].unique()
                    logger.warning(f"⚠️ FILTERING OUT {len(non_priority)} recommendations to non-priority shops: {non_priority_shops}")
                    status_text.text(f"🔍 Filtering to priority shops only...")
                    recommendations = recommendations[recommendations['Destination Shop'].isin(Config.PRIORITY_SHOPS)]
                    logger.info(f"✅ Filtered: {initial_count} → {len(recommendations)} recommendations (priority shops only)")
                    st.info(f"ℹ️ Filtered to {len(recommendations)} recommendations (Priority Shops: {', '.join(Config.PRIORITY_SHOPS)})")
                else:
                    logger.info(f"✅ All {len(recommendations)} recommendations already to priority shops")
            
            # Show info message based on checkbox state
            if generate_all:
                st.success(f"✅ Generated ALL {len(recommendations):,} recommendations without any limit!")
            elif limited_load and len(recommendations) >= 50000:
                st.info(f"ℹ️ Showing first {len(recommendations):,} recommendations. Check '📊 Generate ALL recommendations' above and regenerate for complete dataset.")
            
            # No need to filter by shop again - already done in optimized query

            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Store recommendations and inventory data in session state
            st.session_state.recommendations = recommendations
            st.session_state.slow_shops = slow_shops
            st.session_state.fast_shops = fast_shops
            st.session_state.page_number = 0  # Reset to first page

        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Recommendation generation error: {e}")
            return

        # Retrieve recommendations from session state if available
        if 'recommendations' in st.session_state:
            recommendations = st.session_state.recommendations
        
        if not recommendations.empty:
            # Format columns only once and store in session state
            if 'final_recs' not in st.session_state or gen_clicked:
                # Recommendations now include Sales_Status via Slow/Fast Moving columns
                recommendations_display = recommendations.copy()
                
                logger.info(f"📊 Starting with {len(recommendations_display)} recommendations")
                
                # If Sales_Status column doesn't exist (from old cache), create it
                if 'Sales_Status' not in recommendations_display.columns:
                    logger.warning(f"⚠️ Sales_Status missing - will merge from inventory_df")
                    
                    # Create a lookup table with unique ITEM_CODE + Source Shop combinations
                    # CRITICAL FIX: Drop duplicates before merge to prevent row multiplication
                    inventory_lookup = inventory_df[['ITEM_CODE', 'SHOP_CODE', 'Sales_Status']].copy()
                    
                    # Log inventory_df stats
                    total_inv = len(inventory_lookup)
                    unique_inv = len(inventory_lookup.drop_duplicates(subset=['ITEM_CODE', 'SHOP_CODE']))
                    logger.info(f"📊 Inventory: {total_inv} rows, {unique_inv} unique (ITEM_CODE, SHOP_CODE)")
                    
                    if total_inv != unique_inv:
                        logger.warning(f"⚠️ Inventory has {total_inv - unique_inv} duplicate ITEM_CODE+SHOP_CODE combinations!")
                        inventory_lookup = inventory_lookup.drop_duplicates(subset=['ITEM_CODE', 'SHOP_CODE'])
                    
                    inventory_lookup = inventory_lookup.rename(columns={'SHOP_CODE': 'Source Shop'})
                    
                    # Merge to get Sales_Status from inventory
                    before_merge = len(recommendations_display)
                    recommendations_display = recommendations_display.merge(
                        inventory_lookup,
                        left_on=['ITEM_CODE', 'Source Shop'],
                        right_on=['ITEM_CODE', 'Source Shop'],
                        how='left'
                    )
                    after_merge = len(recommendations_display)
                    
                    recommendations_display['Sales_Status'] = recommendations_display['Sales_Status'].fillna('Slow')
                    
                    logger.info(f"✅ Merged Sales_Status: {before_merge} → {after_merge} rows")
                    
                    if after_merge != before_merge:
                        logger.error(f"❌ ROW COUNT MISMATCH! Merge created {after_merge - before_merge} duplicates!")
                        logger.error(f"❌ {before_merge} → {after_merge} (difference: {after_merge - before_merge})")
                        
                        # Emergency fix: drop exact duplicates
                        recommendations_display = recommendations_display.drop_duplicates()
                        logger.info(f"🔧 After drop_duplicates: {len(recommendations_display)} rows")
                else:
                    logger.info(f"✅ Sales_Status already exists - no merge needed")
                
                # Define column order with all new columns
                preferred_cols = [
                    'ITEM_CODE', 'Item Name', 'Group', 'Sub Group', 'Sales_Status',
                    'Source Shop', 'Source Stock', 'Source Last 30d Sales', 
                    'Source Slow Moving', 'Source Fast Moving',
                    'Source Last GRN Date', 'Source Last WH GRN Date', 'Source GRN Age (days)',
                    'Source Expiry Date', 'Source Expiry Days',
                    'Destination Shop', 'Destination Stock', 
                    'Destination Last 30d Sales',
                    'Destination Last WH GRN Date', 'WH GRN +30 Date', 'Destination Sales (WH GRN +30d)',
                    'Destination Sales Used', 'Sales Source',
                    'Destination Last GRN Date', 'Destination GRN Age (days)',
                    'Recommended_Qty',
                    'Destination Updated Stock', 'Destination Final Stock In Hand Days',
                    'Remark'
                ]
                final_recs = recommendations_display[[c for c in preferred_cols if c in recommendations_display.columns]].copy()

                if not show_grn_info:
                    final_recs = final_recs.drop(columns=['Destination GRN Sales', 'GRN Date', 'Source Last GRN Date', 'Source GRN Age (days)', 'Destination Last GRN Date', 'Destination GRN Age (days)'], errors='ignore')
                
                logger.info(f"📊 Final display: {len(final_recs)} rows")
                st.session_state.final_recs = final_recs
                status_text.empty()
                progress_bar.empty()
            else:
                final_recs = st.session_state.final_recs
                logger.info(f"📊 Using cached final_recs: {len(final_recs)} rows")

            st.subheader(f"🚚 {len(final_recs):,} Transfer Recommendations")
            grn_status = "ON" if use_grn_logic else "OFF"
            st.caption(f"GRN: {grn_status} | Sales: {start_date.strftime('%m-%d')} to {end_date.strftime('%m-%d')} | 📊 Slow Moving items (Sales < {threshold}) are primary transfer candidates")
            
            # Warning for large datasets
            if len(final_recs) > 1000000:
                st.warning(f"⚠️ Large dataset: {len(final_recs):,} rows. Excel download will be truncated to ~1M rows. Use CSV for complete data.")
            elif len(final_recs) > 500000:
                st.info(f"ℹ️ Large dataset: {len(final_recs):,} rows. Consider using filters to reduce size.")

            # Initialize session state for pagination
            if 'page_number' not in st.session_state:
                st.session_state.page_number = 0
            if 'rows_per_page' not in st.session_state:
                st.session_state.rows_per_page = 1000
            
            # Calculate pagination
            total_rows = len(final_recs)
            rows_per_page = st.session_state.rows_per_page
            total_pages = (total_rows + rows_per_page - 1) // rows_per_page
            start_idx = st.session_state.page_number * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            # Add CSS to ensure scrollbars are visible
            st.markdown("""
                <style>
                    /* Force scrollbars to be always visible */
                    div[data-testid="stDataFrame"] > div {
                        overflow: auto !important;
                    }
                    div[data-testid="stDataFrame"] > div > div {
                        overflow-x: scroll !important;
                        overflow-y: scroll !important;
                    }
                    /* Make scrollbars more visible */
                    div[data-testid="stDataFrame"]::-webkit-scrollbar {
                        width: 12px !important;
                        height: 12px !important;
                    }
                    div[data-testid="stDataFrame"]::-webkit-scrollbar-track {
                        background: #f1f1f1 !important;
                    }
                    div[data-testid="stDataFrame"]::-webkit-scrollbar-thumb {
                        background: #888 !important;
                        border-radius: 6px !important;
                    }
                    div[data-testid="stDataFrame"]::-webkit-scrollbar-thumb:hover {
                        background: #555 !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Prepare dataframe slice for display with expiry status indicator
            display_df = final_recs.iloc[start_idx:end_idx].copy()
            
            # Add visual expiry status indicator column if Source Expiry Date exists
            if 'Source Expiry Date' in display_df.columns and 'Source Last GRN Date' in display_df.columns:
                def get_expiry_status(row):
                    expiry_date = row.get('Source Expiry Date')
                    expiry_days = row.get('Source Expiry Days')
                    
                    # Check if should skip (null, blank, or default dates)
                    if pd.isna(expiry_date) or expiry_date == '' or expiry_date == 0:
                        return ''
                    if str(expiry_date) in ['1900-00-00 00:00:00', '1900-01-01 00:00:00', '1900-01-01', '1900-01-01 00:00:00.000000']:
                        return ''
                    
                    # Check expiry_days for status
                    if pd.notna(expiry_days):
                        try:
                            days = float(expiry_days)
                            
                            # Red if negative (already expired)
                            if days < 0:
                                return '🔴 EXPIRED'
                            # Yellow if within 30 days
                            elif 0 <= days <= 30:
                                return '🟡 EXPIRING'
                            # Green if beyond 30 days
                            else:
                                return '🟢 OK'
                        except:
                            return ''
                    return ''
                
                # Add status column after Source Expiry Days
                display_df.insert(
                    display_df.columns.get_loc('Source Expiry Days') + 1 if 'Source Expiry Days' in display_df.columns else len(display_df.columns),
                    'Expiry Status',
                    display_df.apply(get_expiry_status, axis=1)
                )
            
            # Display dataframe with compact widths for better scrolling
            col_config = {}
            for col in display_df.columns:
                if col == 'ITEM_CODE':
                    col_config[col] = st.column_config.TextColumn(col, width=70)
                elif col in ['Source Shop', 'Destination Shop']:
                    col_config[col] = st.column_config.TextColumn(col, width=85)
                elif col == 'Sales_Status':
                    col_config[col] = st.column_config.TextColumn(col, width=75)
                elif col == 'Expiry Status':
                    col_config[col] = st.column_config.TextColumn(col, width=100)
                elif col == 'Item Name':
                    col_config[col] = st.column_config.TextColumn(col, width=250)
                elif col == 'Remark':
                    col_config[col] = st.column_config.TextColumn(col, width=200)
                elif col in ['Group', 'Sub Group']:
                    col_config[col] = st.column_config.TextColumn(col, width=90)
                elif 'Moving' in col:
                    col_config[col] = st.column_config.TextColumn(col, width=70)
                elif 'Stock' in col or 'Qty' in col:
                    col_config[col] = st.column_config.NumberColumn(col, width=100)
                elif 'Sales' in col or 'Age' in col or 'Days' in col:
                    col_config[col] = st.column_config.NumberColumn(col, width=110)
                elif 'Expiry Date' in col:
                    col_config[col] = st.column_config.DatetimeColumn(col, width=120, format="YYYY-MM-DD")
                elif 'Date' in col:
                    col_config[col] = st.column_config.Column(col, width=110)
                else:
                    col_config[col] = st.column_config.Column(col, width=80)
            
            # Display with proper scrolling - use container width to avoid double scrollbars
            st.dataframe(
                display_df, 
                height=600,
                hide_index=True,
                column_config=col_config,
                use_container_width=True
            )
            
            # Pagination controls - right aligned below table with colored icons
            col_space, col_pagination = st.columns([3, 1])
            with col_pagination:
                # Custom CSS for icon-only colored buttons
                st.markdown("""
                    <style>
                    .pagination-container {
                        display: flex;
                        align-items: center;
                        justify-content: flex-end;
                        gap: 10px;
                        padding: 10px 0;
                    }
                    .page-info {
                        font-size: 14px;
                        color: #666;
                        margin-right: 10px;
                    }
                    .nav-btn {
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        font-size: 18px;
                        border: none;
                    }
                    .nav-btn.enabled {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    }
                    .nav-btn.enabled:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                    }
                    .nav-btn.disabled {
                        background: transparent;
                        color: #ccc;
                        cursor: not-allowed;
                    }
                    .nav-separator {
                        font-size: 18px;
                        color: #999;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div class='pagination-container'>
                        <span class='page-info'>Page {st.session_state.page_number + 1}/{total_pages}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col_prev, col_sep, col_next = st.columns([1, 0.3, 1])
                with col_prev:
                    if st.button("◀", disabled=(st.session_state.page_number == 0), key="prev_btn"):
                        st.session_state.page_number -= 1
                        st.rerun()
                with col_sep:
                    st.markdown("<div style='text-align: center; font-size: 18px; color: #999; padding-top: 8px;'>|</div>", unsafe_allow_html=True)
                with col_next:
                    if st.button("▶", disabled=(st.session_state.page_number >= total_pages - 1), key="next_btn"):
                        st.session_state.page_number += 1
                        st.rerun()

            # Download buttons - stack on mobile
            col1, col2 = st.columns([1, 1])
            with col1:
                # For CSV download, regenerate all recommendations without limit if lazy loading was applied
                if limited_load and len(recommendations) >= 5000:
                    # Show button to generate complete dataset
                    if st.button("🔄 Generate Complete Data for Download", key="gen_complete_btn", use_container_width=True):
                        with st.spinner("Generating complete dataset..."):
                            complete_recs = cached_generate_recommendations(
                                group=group,
                                subgroup=subgroup,
                                product=product,
                                shop=shop,
                                item_type=item_type,
                                supplier=supplier,
                                item_name=item_name,
                                use_grn_logic=use_grn_logic,
                                threshold=threshold,
                                use_cache=False,  # Skip cache for complete dataset
                                limit=None,  # No limit - get all data
                                use_parallel=True
                            )
                            st.session_state.complete_csv_data = convert_to_csv(complete_recs)
                            st.success(f"✅ Complete dataset ready: {len(complete_recs):,} recommendations")
                    
                    # Show download button only after complete data is generated
                    if 'complete_csv_data' in st.session_state:
                        st.download_button(
                            "📥 Download Complete CSV",
                            st.session_state.complete_csv_data,
                            f"recommendations_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key="rec_csv_complete",
                            use_container_width=True
                        )
                else:
                    # For small datasets (<50k rows), offer XLSX with formatting
                    if len(final_recs) < 50000:
                        # Cache XLSX generation for smaller datasets
                        if 'xlsx_data' not in st.session_state or gen_clicked:
                            st.session_state.xlsx_data = convert_recommendations_to_xlsx(final_recs)
                        
                        st.download_button(
                            "📥 Download XLSX (with colors)",
                            st.session_state.xlsx_data,
                            f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="rec_xlsx",
                            use_container_width=True
                        )
                    else:
                        # Cache CSV generation for larger datasets
                        if 'csv_data' not in st.session_state or gen_clicked:
                            st.session_state.csv_data = convert_to_csv(final_recs)
                        
                        st.download_button(
                            "📥 Download CSV",
                            st.session_state.csv_data,
                            f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key="rec_csv",
                            use_container_width=True
                        )
            with col2:
                if len(final_recs) > 1000000:
                    st.warning(f"⚠️ {len(final_recs):,} rows (Excel limited to ~1M)")
                else:
                    st.success(f"✅ {len(final_recs):,} recommendations ready")
        else:
            st.info("ℹ️ No recommendations for current filters")

        # Clear progress indicators if they exist
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

        # Cache Excel generation
        if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
            filters_used = {
                "Groups": group,
                "Sub Group": subgroup,
                "Product": product,
                "Shop": shop,               
                "Threshold": threshold,
                "GRN Logic": use_grn_logic,
                "Sales Period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }

            if 'excel_data' not in st.session_state or gen_clicked:
                final_recs = st.session_state.final_recs
                st.session_state.excel_data = convert_to_excel(slow_shops, fast_shops, final_recs, filters_used)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            st.download_button(
                "📊 Download Excel Report (All Sheets)",
                st.session_state.excel_data,
                f"Inventory_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="rec_excel",
                use_container_width=True
            )

if __name__ == "__main__":
    main()