"""
cleaner.py — Data Cleaner Skill
Handles: missing values, duplicates, type fixing, whitespace, outliers
"""

import pandas as pd
import numpy as np
import os


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Full cleaning pipeline.
    Returns (cleaned_df, list_of_actions_taken)
    """
    steps = []
    df = df.copy()

    # 1. Strip whitespace from column names
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    if df.columns.tolist() != original_cols:
        steps.append("✅ Column names cleaned (whitespace stripped, spaces → underscores)")

    # 2. Remove fully duplicate rows
    dups = df.duplicated().sum()
    if dups > 0:
        df = df.drop_duplicates()
        steps.append(f"✅ Removed {dups} duplicate rows")

    # 3. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    if len(str_cols) > 0:
        steps.append(f"✅ Stripped whitespace from {len(str_cols)} string columns")

    # 4. Fix numeric columns stored as strings
    for col in str_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = converted.notna().sum() / max(len(df), 1)
        if non_null_ratio > 0.8:
            df[col] = converted
            steps.append(f"✅ Converted '{col}' to numeric")

    # 5. Fix date columns
    date_keywords = ["date", "time", "year", "month", "day"]
    for col in df.select_dtypes(include="object").columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                steps.append(f"✅ Converted '{col}' to datetime")
            except Exception:
                pass

    # 6. Handle missing values
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            steps.append(f"✅ Filled {missing[col]} missing values in '{col}' with median ({fill_val:.2f})")
        else:
            fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            steps.append(f"✅ Filled {missing[col]} missing values in '{col}' with mode ('{fill_val}')")

    # 7. Remove outliers (IQR method) from numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    removed_total = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            df = df[~outlier_mask]
            removed_total += n_outliers
    if removed_total > 0:
        steps.append(f"✅ Removed {removed_total} extreme outlier rows (IQR × 3 method)")

    if not steps:
        steps.append("✅ Data was already clean — no changes needed")

    return df, steps


def save_original(df: pd.DataFrame, filename: str, folder: str = "data/original") -> str:
    """Save the original (uncleaned) DataFrame as CSV."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    return path


def save_cleaned(df: pd.DataFrame, filename: str, folder: str = "data/cleaned") -> str:
    """Save the cleaned DataFrame as CSV."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    return path


def get_cleaning_summary(original: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Compare original vs cleaned DataFrames."""
    return {
        "original_rows": len(original),
        "cleaned_rows": len(cleaned),
        "rows_removed": len(original) - len(cleaned),
        "original_columns": len(original.columns),
        "cleaned_columns": len(cleaned.columns),
        "original_missing": int(original.isnull().sum().sum()),
        "cleaned_missing": int(cleaned.isnull().sum().sum()),
    }
