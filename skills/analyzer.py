"""
analyzer.py — AI Analyst Skill
Handles: EDA, statistics, insights, business questions, dashboard suggestions
Uses OpenAI API for AI-powered analysis
"""

import pandas as pd
import numpy as np
import json
import os
from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")
    return OpenAI(api_key=api_key)


# ──────────────────────────────────────────────
# LOCAL STATISTICAL ANALYSIS (no API needed)
# ──────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> dict:
    """Return descriptive statistics for all columns."""
    stats = {}
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats["numeric"] = numeric_df.describe().round(2).to_dict()

    cat_df = df.select_dtypes(include="object")
    if not cat_df.empty:
        cat_stats = {}
        for col in cat_df.columns:
            vc = df[col].value_counts()
            cat_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "top_5": vc.head(5).to_dict(),
            }
        stats["categorical"] = cat_stats

    return stats


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(3)


def top_correlations(df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Return pairs of columns with high correlation."""
    corr = correlation_analysis(df)
    if corr.empty:
        return []
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(val), 3),
                    "strength": "Strong" if abs(val) >= 0.75 else "Moderate",
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def quick_insights(df: pd.DataFrame) -> list[str]:
    """Generate rule-based insights without AI."""
    insights = []
    total = len(df)

    # Missing data
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(1)
    high_missing = missing_pct[missing_pct > 10]
    if not high_missing.empty:
        for col, pct in high_missing.items():
            insights.append(f"⚠️ Column '{col}' has {pct}% missing values — consider review")

    # Numeric distribution
    for col in df.select_dtypes(include=[np.number]).columns:
        skew = df[col].skew()
        if abs(skew) > 1.5:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            insights.append(f"📊 '{col}' is heavily skewed {direction} (skew={skew:.2f})")

    # Categorical cardinality
    for col in df.select_dtypes(include="object").columns:
        n_unique = df[col].nunique()
        if n_unique == total:
            insights.append(f"🔑 '{col}' appears to be a unique identifier (all values distinct)")
        elif n_unique == 1:
            insights.append(f"⚠️ '{col}' has only one unique value — may not be useful for analysis")
        elif n_unique <= 10:
            top = df[col].value_counts().index[0]
            pct = round(df[col].value_counts().iloc[0] / total * 100, 1)
            insights.append(f"📌 '{col}' top category: '{top}' ({pct}% of data)")

    # Correlation
    corr_pairs = top_correlations(df, threshold=0.7)
    for pair in corr_pairs[:3]:
        insights.append(
            f"🔗 Strong correlation ({pair['correlation']}) between '{pair['col_a']}' and '{pair['col_b']}'"
        )

    return insights if insights else ["✅ No major data issues detected from quick scan"]


# ──────────────────────────────────────────────
# AI-POWERED ANALYSIS (OpenAI)
# ──────────────────────────────────────────────

def _build_data_context(df: pd.DataFrame) -> str:
    """Build a compact text summary of the DataFrame for the AI prompt."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    lines = [
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}",
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}",
        f"Date columns ({len(date_cols)}): {', '.join(date_cols)}",
        f"Missing values: {int(df.isnull().sum().sum())} total",
        "",
        "Sample statistics (numeric):",
    ]

    if numeric_cols:
        stats = df[numeric_cols[:6]].describe().round(2).to_string()
        lines.append(stats)

    if cat_cols:
        lines.append("\nTop categories:")
        for col in cat_cols[:4]:
            top = df[col].value_counts().head(3).to_dict()
            lines.append(f"  {col}: {top}")

    lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string(index=False)}")

    return "\n".join(lines)


def ai_insights(df: pd.DataFrame, model: str = "gpt-4o-mini") -> str:
    """Use OpenAI to generate business insights from the data."""
    client = get_openai_client()
    context = _build_data_context(df)

    prompt = f"""You are a senior data analyst. Analyze this dataset and provide:

1. KEY INSIGHTS (5-7 bullet points) — most important findings
2. TRENDS & PATTERNS — what the data reveals over time or across categories
3. ANOMALIES — anything unusual or worth investigating
4. BUSINESS RECOMMENDATIONS — 3-5 actionable suggestions based on the data

Dataset summary:
{context}

Be specific, use numbers from the data, and focus on business value."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content


def ai_answer_question(df: pd.DataFrame, question: str, model: str = "gpt-4o-mini") -> str:
    """Answer a specific business question about the data using AI."""
    client = get_openai_client()
    context = _build_data_context(df)

    # Also compute a quick numeric answer if possible
    numeric_summary = df.select_dtypes(include=[np.number]).agg(
        ["sum", "mean", "min", "max"]
    ).round(2).to_string() if not df.select_dtypes(include=[np.number]).empty else "No numeric columns"

    prompt = f"""You are a data analyst. A user has uploaded a dataset and asked a question.
Answer the question accurately using the data provided. Show calculations where relevant.

Dataset summary:
{context}

Numeric aggregates:
{numeric_summary}

User question: {question}

Provide a clear, direct answer with supporting evidence from the data."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


def ai_dashboard_suggestions(df: pd.DataFrame, model: str = "gpt-4o-mini") -> str:
    """Ask AI to suggest dashboard charts and KPIs."""
    client = get_openai_client()
    context = _build_data_context(df)

    prompt = f"""You are a BI developer and data visualization expert.
Based on the dataset below, suggest a complete dashboard design.

For each suggestion include:
- Chart type (bar, line, pie, scatter, heatmap, etc.)
- X-axis and Y-axis (or dimensions)
- Why this chart is useful
- What business question it answers

Also suggest 4-6 KPI cards (key metrics to show at the top of the dashboard).

Dataset summary:
{context}

Format your response clearly with sections for: KPI Cards, Charts, and Filters."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1000,
    )
    return response.choices[0].message.content
