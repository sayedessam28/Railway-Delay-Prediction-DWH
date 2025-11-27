#!/usr/bin/env python3
"""
etl_railway_pro.py
Complete, production-minded ETL for railway.csv + railway_data_dictionary.csv

Features:
- Modular functions for Extract / Transform / Load
- Thorough transformations with robust imputation and categorical cleaning
- Data quality validations & report
- Load to SQL Server Staging Table with dynamic schema creation
- CLI flags: --dry-run (default), --load (perform DB load), --save (save cleaned CSV)
"""

import os
import sys
import argparse
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import datetime
import urllib
from io import StringIO

# Optional DB libs
try:
    from sqlalchemy import create_engine, text, event
    from sqlalchemy.types import String, DateTime, Time, Integer, Float
except Exception:
    create_engine = None
    text = None
    event = None
    class String: pass
    class DateTime: pass
    class Time: pass
    class Integer: pass
    class Float: pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------
# CONFIG - edit these
# -----------------------
CSV_FILENAME = "railway.csv"
DATA_DICT_FILENAME = "railway_data_dictionary.csv"
CLEANED_OUT = "railway_clean_etl.csv"

# DB config
DB_DRIVER = "ODBC Driver 17 for SQL Server"
DB_SERVER = os.getenv("RAIL_DB_SERVER", "Sayed")
DB_NAME = os.getenv("RAIL_DB_NAME", "RailwayDW")
DB_USER = os.getenv("RAIL_DB_USER", "")
DB_PASS = os.getenv("RAIL_DB_PASS", "")
USE_TRUSTED = os.getenv("RAIL_DB_TRUSTED", "True").lower() in ("1","true","yes")

STAGING_TABLE = "stg_railway_tickets"
TO_SQL_CHUNKSIZE = 2000

# -----------------------
# UTILS
# -----------------------
def assert_file_exists(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# -----------------------
# EXTRACT
# -----------------------
def extract(csv_path: str) -> pd.DataFrame:
    assert_file_exists(csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=["", "NA", "N/A", "nan"])
    logging.info("Extracted CSV rows=%d cols=%d", df.shape[0], df.shape[1])
    return df

def load_data_dictionary(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        dd = pd.read_csv(path, dtype=str)
        logging.info("Loaded data dictionary rows=%d", dd.shape[0])
        return dd
    logging.warning("Data dictionary not found at %s", path)
    return pd.DataFrame()

# -----------------------
# TRANSFORM - small composable steps
# -----------------------
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip()
                     .str.lower()
                     .str.replace(" ", "_", regex=False)
                     .str.replace("-", "_", regex=False))
    return df

def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].where(df[c].isnull(), df[c].astype(str).str.strip())
    return df

def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return df

def normalize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["purchase_type", "payment_method", "railcard", "ticket_class", "ticket_type", "journey_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({"nan": "missing", "none": "missing"}).fillna("missing")
            
            if col == "railcard":
                df[col] = df[col].replace({"missing": "none"})
                
            if col == "ticket_class":
                df[col] = df[col].replace({"first": "first class", "standard": "standard class"})
                
            if col == "journey_status":
                df[col] = df[col].str.title().replace({"Missing": "Unknown"})
            
    return df


def parse_dates_and_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["date_of_purchase", "date_of_journey"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["time_of_purchase", "departure_time", "arrival_time", "actual_arrival_time"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.isna().all():
                 parsed = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce")
                 if parsed.isna().all():
                     parsed = pd.to_datetime(df[col], format="%H:%M", errors="coerce")
            df[col] = parsed.dt.time
    return df

def fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype('Int64')
    
    if "reason_for_delay" in df.columns and "journey_status" in df.columns:
        df["reason_for_delay"] = df["reason_for_delay"].astype(str).str.strip().str.title().replace({"Nan": np.nan, "None": np.nan, "Missing": np.nan})
        
        status = df["journey_status"].astype(str).str.strip()
        
        on_time_mask = status == 'On Time'
        df.loc[on_time_mask, "reason_for_delay"] = "On Time"
        
        cancelled_mask = status == 'Cancelled'
        df.loc[cancelled_mask & df["reason_for_delay"].isnull(), "reason_for_delay"] = "Cancellation"
        
        df["reason_for_delay"] = df["reason_for_delay"].fillna("Unknown/Missing Reason")


    if "actual_arrival_time" in df.columns and "arrival_time" in df.columns and "journey_status" in df.columns:
        status = df["journey_status"].astype(str).str.strip()
        on_time_mask = status == 'On Time'
        is_actual_missing = pd.isnull(df["actual_arrival_time"])
        
        df.loc[on_time_mask & is_actual_missing, "actual_arrival_time"] = df["arrival_time"]
    
    if "refund_request" in df.columns:
        df["refund_request"] = df["refund_request"].astype(str).str.strip().str.title().replace({"Y":"Yes","N":"No","True":"Yes","False":"No", "Nan": "No", "None": "No"})
        df["refund_request"] = df["refund_request"].fillna("No")

    return df

def combine_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def combine(d, t):
        try:
            if pd.notnull(d) and pd.notnull(t):
                return pd.Timestamp.combine(d.date(), t)
        except Exception:
            return pd.NaT
        return pd.NaT
        
    if "date_of_purchase" in df.columns and "time_of_purchase" in df.columns:
        df["purchase_datetime"] = df.apply(lambda r: combine(r["date_of_purchase"], r["time_of_purchase"]), axis=1)
        
    if "date_of_journey" in df.columns and "departure_time" in df.columns:
        df["departure_datetime"] = df.apply(lambda r: combine(r["date_of_journey"], r["departure_time"]), axis=1)
        
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # ----------------------------------------------------
    # 1. Performance Measures: Scheduled Duration and Delay
    # ----------------------------------------------------
    if set(["departure_time", "arrival_time"]).issubset(df.columns):
        def compute_duration(dep, arr):
            try:
                if pd.isnull(dep) or pd.isnull(arr):
                    return np.nan
                a = datetime.datetime.combine(datetime.date(2000,1,1), arr)
                b = datetime.datetime.combine(datetime.date(2000,1,1), dep)
                
                diff = (a - b).total_seconds() / 60.0
                if diff < 0:
                    diff += 24*60 
                return diff
            except Exception:
                return np.nan
        
        if set(["departure_time", "actual_arrival_time"]).issubset(df.columns):
            df["duration_minutes"] = df.apply(lambda r: compute_duration(r["departure_time"], r["actual_arrival_time"]), axis=1).round(2)

        df["scheduled_duration_minutes"] = df.apply(
            lambda r: compute_duration(r["departure_time"], r["arrival_time"]), axis=1
        ).round(2)
        
        df["delay_minutes"] = (df["duration_minutes"] - df["scheduled_duration_minutes"]).round(2)
        df["delay_minutes"] = df["delay_minutes"].apply(lambda x: max(0, x))


    # ----------------------------------------------------
    # 2. Price Measures: Standard Price and Discount Value
    # ----------------------------------------------------
    if "price" in df.columns:
        
        def calculate_standard_price(row):
            price = row['price']
            ticket_type = str(row['ticket_type']).lower()
            railcard = str(row['railcard']).lower()
            
            if pd.isnull(price) or price == 0:
                return 0

            type_factor = 1.0
            if 'advance' in ticket_type:
                type_factor = 0.5
            elif 'off-peak' in ticket_type:
                type_factor = 0.75
                
            rail_factor = 1.0
            if railcard not in ['none', 'missing']:
                rail_factor = 0.6666666666666667
                
            total_discount_factor = type_factor * rail_factor
            
            if total_discount_factor == 0:
                return price
                
            return round(price / total_discount_factor)

        df["standard_price"] = df.apply(calculate_standard_price, axis=1).astype('Int64')
        
        df["discount_value"] = (df["standard_price"] - df["price"]).astype('Int64')


    # ----------------------------------------------------
    # 3. Boolean and Other Features
    # ----------------------------------------------------
    if "journey_status" in df.columns:
        status = df["journey_status"].astype(str).str.lower()
        df["is_delayed"] = status.apply(lambda x: 1 if "delayed" in x else 0).astype('Int64')
        df["is_cancelled"] = status.apply(lambda x: 1 if "cancelled" in x else 0).astype('Int64')
    
    if "date_of_journey" in df.columns:
        df["journey_day"] = df["date_of_journey"].dt.day.astype('Int64')
        df["journey_month"] = df["date_of_journey"].dt.month.astype('Int64')
        df["journey_year"] = df["date_of_journey"].dt.year.astype('Int64')
        df["journey_weekday"] = (df["date_of_journey"].dt.weekday + 1).astype('Int64')
        
    if "departure_datetime" in df.columns:
        df["departure_hour"] = df["departure_datetime"].dt.hour.astype('Int64')
        
    if "refund_request" in df.columns:
        df["is_refund_requested"] = df["refund_request"].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == 'yes' else 0).astype('Int64')
        
    return df

def convert_time_objects_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Converts datetime.time objects in specific columns to string format for pyodbc compatibility."""
    df = df.copy()
    time_cols = ["time_of_purchase", "departure_time", "arrival_time", "actual_arrival_time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = np.where(pd.notna(df[col]), df[col].astype(str), df[col])
    logging.info("Converted time objects to strings for pyodbc compatibility.")
    return df

def cleanup_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["departure_station", "arrival_destination"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title().replace({"Nan": np.nan, "None": np.nan, "Missing": np.nan})
            
    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype(str).str.strip()
        
    before = len(df)
    df = df.drop_duplicates()
    logging.info("Dropped %d duplicate rows", before - len(df))
    return df

# Full transform pipeline
def transform(df: pd.DataFrame) -> pd.DataFrame:
    steps = [
        standardize_column_names,
        trim_whitespace,
        normalize_nulls,
        normalize_categorical,
        parse_dates_and_times,
        fill_defaults,
        combine_datetime_columns,
        feature_engineering,
        cleanup_and_dedupe
    ]
    for fn in steps:
        df = fn(df)
    logging.info("Transformation finished: rows=%d cols=%d", df.shape[0], df.shape[1])
    return df

# -----------------------
# VALIDATION / TESTS
# -----------------------
def validate(df: pd.DataFrame) -> Dict[str, Any]:
    checks = {}
    checks['rows'] = len(df)
    checks['cols'] = list(df.columns)
    critical = ["transaction_id", "purchase_datetime", "departure_datetime", "price"]
    for c in critical:
        if c in df.columns:
            checks[f"missing_{c}"] = int(df[c].isnull().sum())
    if "transaction_id" in df.columns:
        checks["transaction_id_duplicates"] = int(df["transaction_id"].duplicated().sum())
    if "price" in df.columns:
        valid_prices = df["price"].dropna()
        checks["price_negative"] = int((valid_prices < 0).sum())
        med = valid_prices.median() if not valid_prices.empty else 0
        checks["price_high_outliers"] = int((valid_prices > med * 10).sum()) if med > 0 else 0
    if "duration_minutes" in df.columns:
        checks["duration_negative"] = int((df["duration_minutes"] < 0).sum())
    if "journey_year" in df.columns:
        bad = df.loc[~df["journey_year"].between(2000,2030, inclusive="both"), "journey_year"].dropna().unique().tolist()
        checks["journey_years_out_of_range_example"] = bad[:5]
    return checks

def fail_on_critical(checks: Dict[str, Any]) -> Tuple[bool, str]:
    if checks.get("missing_transaction_id", 0) > 0:
        return True, "transaction_id missing values found"
    if checks.get("transaction_id_duplicates", 0) > 0:
        return True, "duplicate transaction_id values found"
    if checks.get("missing_purchase_datetime", 0) > 0 or checks.get("missing_departure_datetime", 0) > 0:
        return True, "missing datetimes in purchase/departure columns"
    return False, ""

# -----------------------
# SAVE / LOAD
# -----------------------
def save_clean_csv(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, index=False)
    logging.info("Saved cleaned CSV to %s", out_path)

def get_sqlalchemy_engine(driver=DB_DRIVER, server=DB_SERVER, database=DB_NAME, trusted=USE_TRUSTED, user=DB_USER, password=DB_PASS):
    if create_engine is None:
        raise RuntimeError("sqlalchemy not installed. Install with: pip install sqlalchemy pyodbc")
    if trusted:
        conn_str = f"Driver={{{driver}}};Server={server};Database={database};Trusted_Connection=yes;"
    else:
        conn_str = f"Driver={{{driver}}};Server={server};Database={database};UID={user};PWD={password};Encrypt=yes;TrustServerCertificate=yes;"
    quoted = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}", fast_executemany=True)
    
    try:
        @event.listens_for(engine, "before_cursor_execute")
        def _enable_fast_executemany(conn, cursor, statement, parameters, context, executemany):
            try:
                if executemany and hasattr(cursor, "fast_executemany"):
                    cursor.fast_executemany = True
            except Exception:
                pass
    except Exception:
        pass
        
    return engine

def create_staging_table_sql(df: pd.DataFrame, table_name: str) -> str:
    """Generates a CREATE TABLE SQL statement for SQL Server based on a DataFrame's schema."""
    
    sql_parts = []
    time_cols = ["time_of_purchase", "departure_time", "arrival_time", "actual_arrival_time"]
    
    for col, dtype in df.dtypes.items():
        sql_type = 'NVARCHAR(255)'
        
        # Mapping based on transformed pandas dtypes
        if col in time_cols:
            sql_type = 'TIME(0)'
        elif dtype.name.startswith('Int64'):
            sql_type = 'INT'
        elif dtype.name.startswith('float'):
            sql_type = 'FLOAT'
        elif dtype.name.startswith('datetime'):
            sql_type = 'DATETIME2'
        elif dtype.name == 'object':
            if col == "transaction_id":
                 sql_type = 'VARCHAR(100)'
            elif col in ["departure_station", "arrival_destination", "reason_for_delay"]:
                 sql_type = 'NVARCHAR(255)'
            else:
                 sql_type = 'NVARCHAR(100)'
        
        # Ensure PRIMARY KEY (transaction_id) is NOT NULL
        nullable = "NULL"
        pk = ""
        
        if col == "transaction_id":
            nullable = "NOT NULL"
            pk = " PRIMARY KEY"

        sql_parts.append(f"{col} {sql_type} {nullable}{pk}")

    columns_sql = ",\n\t\t\t\t".join(sql_parts)

    return f"""
        IF OBJECT_ID(N'{table_name}', N'U') IS NULL
        BEGIN
            CREATE TABLE {table_name} (
                {columns_sql}
            );
        END
        """

def create_db_and_staging(engine, df: pd.DataFrame):
    """Creates the staging table dynamically based on the DataFrame schema."""
    if text is None:
          logging.error("SQLAlchemy text function not available. Skipping table creation.")
          return
          
    staging_sql = create_staging_table_sql(df, STAGING_TABLE)
    with engine.begin() as conn:
        conn.execute(text(staging_sql))
    logging.info("Created or ensured staging table exists with dynamic schema.")


def load_to_sql(engine, df: pd.DataFrame, table_name: str, chunksize:int=TO_SQL_CHUNKSIZE):
    """Loads DataFrame to SQL Server using optimized methods."""
    logging.info("Loading to SQL table %s (rows=%d)", table_name, len(df))
    # FIX APPLIED: Removed method="multi" to avoid pyodbc parameter binding error
    df.to_sql(name=table_name, con=engine, if_exists="append", index=False, chunksize=chunksize)
    logging.info("Load completed")

# -----------------------
# ORCHESTRATION 
# -----------------------
def run_pipeline(csv_path: str, dict_path: str, do_save: bool = True, do_load: bool = False):
    raw = extract(csv_path)
    dd = load_data_dictionary(dict_path)
    transformed = transform(raw)
    
    checks = validate(transformed)
    is_critical_fail, reason = fail_on_critical(checks)

    if do_load:
        df_for_sql_load = convert_time_objects_to_strings(transformed)
        
        if create_engine is None:
             logging.error("Cannot load to SQL: sqlalchemy or pyodbc not installed.")
             sys.exit(3)
             
        engine = get_sqlalchemy_engine()
        try:
             with engine.connect() as conn:
                 conn.execute(text("SELECT 1"))
             logging.info("Successfully connected to SQL Server.")
        except Exception as e:
             logging.error("Pipeline failed: %s", e)
             sys.exit(4)

        # 1. Create/Update Staging Table schema
        create_db_and_staging(engine, df_for_sql_load)
        
        # 2. Load the data
        load_to_sql(engine, df_for_sql_load, STAGING_TABLE)

    if do_save:
        save_clean_csv(transformed, CLEANED_OUT)
        
    if is_critical_fail:
        logging.error("Validation failed - Critical issue: %s", reason)
        logging.info("Validation checks: %s", checks)
        raise RuntimeError("ETL Failed validation.")

    logging.info("Validation passed - pipeline safe to load")
    return transformed, checks

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="ETL for Railway dataset")
    parser.add_argument("--csv", default=CSV_FILENAME, help="Path to railway CSV")
    parser.add_argument("--dict", default=DATA_DICT_FILENAME, help="Path to data dictionary CSV")
    parser.add_argument("--no-save", action="store_true", help="Do not save cleaned CSV")
    parser.add_argument("--load", action="store_true", help="Load cleaned data to SQL Server (ensure DB config)")
    args = parser.parse_args()
    try:
        df, checks = run_pipeline(args.csv, args.dict, do_save=not args.no_save, do_load=args.load)
        logging.info("Pipeline finished successfully. Rows=%d", df.shape[0])
        print("Summary checks:", checks)
    except Exception as e:
        # logging.exception is already called inside run_pipeline failure handling
        sys.exit(1)

if __name__ == "__main__":
    main()