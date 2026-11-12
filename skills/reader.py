"""
reader.py — Data Reader Skill
Supports: CSV, Excel, SQLite/SQL, JSON
"""

import pandas as pd
import sqlite3
import os


def read_csv(filepath: str) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(filepath)


def read_excel(filepath: str, sheet_name=0) -> pd.DataFrame:
    """Read an Excel file (.xlsx / .xls) and return a DataFrame."""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def read_json(filepath: str) -> pd.DataFrame:
    """Read a JSON file and return a DataFrame."""
    return pd.read_json(filepath)


def read_sql(db_path: str, query: str = None) -> pd.DataFrame:
    """
    Connect to a SQLite database and execute a query.
    If no query is given, lists available tables and reads the first one.
    """
    conn = sqlite3.connect(db_path)
    if query is None:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if tables.empty:
            conn.close()
            raise ValueError("No tables found in the database.")
        first_table = tables.iloc[0]["name"]
        df = pd.read_sql(f"SELECT * FROM [{first_table}]", conn)
    else:
        df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_sql_tables(db_path: str) -> list:
    """Return list of table names in a SQLite database."""
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    conn.close()
    return tables["name"].tolist()


def read_file(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Auto-detect file type and read accordingly.
    Supported: .csv, .xlsx, .xls, .json, .db, .sqlite
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".csv":
        return read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        sheet = kwargs.get("sheet_name", 0)
        return read_excel(filepath, sheet_name=sheet)
    elif ext == ".json":
        return read_json(filepath)
    elif ext in (".db", ".sqlite"):
        query = kwargs.get("query", None)
        return read_sql(filepath, query=query)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_basic_info(df: pd.DataFrame) -> dict:
    """Return basic metadata about a DataFrame."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample": df.head(5).to_dict(orient="records"),
    }
