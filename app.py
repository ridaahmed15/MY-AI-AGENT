"""
app.py — Smart Data Analyst AI Agent
Main Streamlit interface
"""

import os
import sys
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from skills.reader import read_file, get_basic_info, get_sql_tables
from skills.cleaner import clean_data, save_original, save_cleaned, get_cleaning_summary
from skills.analyzer import (
    descriptive_stats,
    quick_insights,
    top_correlations,
    ai_insights,
    ai_answer_question,
    ai_dashboard_suggestions,
)
from skills.report import generate_report, save_qa_log

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Data Analyst AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f4ff, #e8eeff);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 3px solid #10b981;
        margin-bottom: 0.4rem;
        font-size: 0.93rem;
    }
    .step-box {
        background: #f0fdf4;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border-left: 3px solid #22c55e;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────────────────
defaults = {
    "original_df": None,
    "cleaned_df": None,
    "cleaning_steps": [],
    "cleaning_summary": {},
    "local_insights": [],
    "ai_insights_text": "",
    "ai_dashboard_text": "",
    "filename": "",
    "qa_log": [],
    "report_path": "",
    "report_text": "",
    "cleaned_csv_path": "",
    "original_csv_path": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────────────────

def has_data() -> bool:
    return st.session_state.cleaned_df is not None


def reset_state():
    for k, v in defaults.items():
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # OpenAI key
    api_key_input = st.text_input(
        "🔑 OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required for AI insights, Q&A, and dashboard suggestions",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    ai_model = st.selectbox(
        "🧠 AI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini is fastest and cheapest",
    )

    st.divider()

    st.markdown("## 📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite"],
        help="Supports CSV, Excel, JSON, SQLite",
    )

    sql_query = ""
    if uploaded_file and uploaded_file.name.endswith((".db", ".sqlite")):
        sql_query = st.text_area(
            "SQL Query (optional)",
            placeholder="SELECT * FROM table_name LIMIT 1000",
            height=80,
        )

    if uploaded_file:
        if st.button("🚀 Load & Analyze", type="primary", use_container_width=True):
            reset_state()
            with st.spinner("Reading data..."):
                try:
                    # Save to temp file
                    suffix = os.path.splitext(uploaded_file.name)[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    kwargs = {}
                    if sql_query.strip():
                        kwargs["query"] = sql_query.strip()

                    raw_df = read_file(tmp_path, **kwargs)
                    st.session_state.filename = uploaded_file.name

                    # Save original
                    orig_path = save_original(
                        raw_df,
                        f"original_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                    )
                    st.session_state.original_csv_path = orig_path
                    st.session_state.original_df = raw_df

                    # Clean
                    cleaned_df, steps = clean_data(raw_df)
                    summary = get_cleaning_summary(raw_df, cleaned_df)
                    clean_path = save_cleaned(
                        cleaned_df,
                        f"cleaned_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                    )
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.cleaning_steps = steps
                    st.session_state.cleaning_summary = summary
                    st.session_state.cleaned_csv_path = clean_path

                    # Local insights
                    st.session_state.local_insights = quick_insights(cleaned_df)

                    os.unlink(tmp_path)
                    st.success("✅ Data loaded and cleaned!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()

    if has_data():
        st.markdown("## 🤖 AI Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 AI Insights", use_container_width=True):
                with st.spinner("Generating AI insights..."):
                    try:
                        result = ai_insights(st.session_state.cleaned_df, model=ai_model)
                        st.session_state.ai_insights_text = result
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"❌ {e}")
        with col2:
            if st.button("📊 Dashboard Ideas", use_container_width=True):
                with st.spinner("Generating dashboard suggestions..."):
                    try:
                        result = ai_dashboard_suggestions(st.session_state.cleaned_df, model=ai_model)
                        st.session_state.ai_dashboard_text = result
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.divider()
        if st.button("📄 Generate Full Report", use_container_width=True, type="primary"):
            with st.spinner("Building report..."):
                try:
                    path, text = generate_report(
                        filename=st.session_state.filename,
                        original_df=st.session_state.original_df,
                        cleaned_df=st.session_state.cleaned_df,
                        cleaning_steps=st.session_state.cleaning_steps,
                        cleaning_summary=st.session_state.cleaning_summary,
                        local_insights=st.session_state.local_insights,
                        ai_insights=st.session_state.ai_insights_text,
                        ai_dashboard=st.session_state.ai_dashboard_text,
                    )
                    st.session_state.report_path = path
                    st.session_state.report_text = text
                    st.success(f"✅ Report saved!")
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.session_state.report_text:
            st.download_button(
                "⬇️ Download Report",
                data=st.session_state.report_text,
                file_name=f"report_{st.session_state.filename}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.session_state.cleaned_csv_path and os.path.exists(st.session_state.cleaned_csv_path):
            with open(st.session_state.cleaned_csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Cleaned CSV",
                    data=f.read(),
                    file_name=f"cleaned_{st.session_state.filename}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🤖 Smart Data Analyst AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any data → Auto-clean → AI Analysis → Insights → Reports</div>', unsafe_allow_html=True)

# ── Empty state ──────────────────────────────────────────────────────────────
if not has_data():
    st.info("👈 Upload a file from the sidebar to get started.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <b>📄 CSV / Excel</b><br>
            <small>Upload .csv, .xlsx, .xls</small>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <b>🗄️ SQL Database</b><br>
            <small>Upload .db or .sqlite files</small>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <b>🧹 Auto Clean</b><br>
            <small>Nulls, duplicates, types</small>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <b>🧠 AI Insights</b><br>
            <small>GPT-powered analysis</small>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data Overview",
    "🧹 Cleaning",
    "📊 EDA & Charts",
    "💡 Insights",
    "❓ Ask AI",
    "📄 Report",
])

df = st.session_state.cleaned_df
orig = st.session_state.original_df


# ════════════════════════════════════════════════════════════
#  TAB 1 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    summary = st.session_state.cleaning_summary
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows (cleaned)", f"{len(df):,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Rows removed", summary.get("rows_removed", 0))
    m4.metric("Missing (after)", summary.get("cleaned_missing", 0))
    m5.metric("Numeric cols", len(df.select_dtypes(include="number").columns))

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("🔍 Sample Data")
        n_rows = st.slider("Rows to show", 5, min(100, len(df)), 10, key="overview_rows")
        st.dataframe(df.head(n_rows), use_container_width=True)

    with col_b:
        st.subheader("📐 Column Info")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Nulls": df.isnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    st.subheader("📊 Missing Values Heatmap")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("✅ No missing values in cleaned dataset!")
    else:
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={"x": "Column", "y": "Missing Count"},
            title="Missing Values per Column",
            color=missing.values,
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — CLEANING
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🧹 Data Cleaning Report</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Before vs After")
        before_after = pd.DataFrame({
            "Metric": ["Rows", "Missing Values", "Duplicate Rows"],
            "Before": [
                summary.get("original_rows", 0),
                summary.get("original_missing", 0),
                summary.get("rows_removed", 0),
            ],
            "After": [
                summary.get("cleaned_rows", 0),
                summary.get("cleaned_missing", 0),
                0,
            ],
        })
        st.dataframe(before_after, hide_index=True, use_container_width=True)

        fig = go.Figure(data=[
            go.Bar(name="Before", x=before_after["Metric"], y=before_after["Before"],
                   marker_color="#ef4444"),
            go.Bar(name="After", x=before_after["Metric"], y=before_after["After"],
                   marker_color="#22c55e"),
        ])
        fig.update_layout(barmode="group", title="Before vs After Cleaning")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Cleaning Steps Taken")
        for step in st.session_state.cleaning_steps:
            st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

    st.subheader("📁 Saved Files")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Original CSV:** `{st.session_state.original_csv_path}`")
    with col2:
        st.info(f"**Cleaned CSV:** `{st.session_state.cleaned_csv_path}`")


# ════════════════════════════════════════════════════════════
#  TAB 3 — EDA & CHARTS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Descriptive stats
    with st.expander("📐 Descriptive Statistics", expanded=True):
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
        else:
            st.write("No numeric columns found.")

    # Distribution plots
    if numeric_cols:
        st.subheader("📈 Distribution Analysis")
        col_sel = st.selectbox("Select numeric column", numeric_cols, key="dist_col")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col_sel, nbins=40, title=f"Distribution of {col_sel}",
                               color_discrete_sequence=["#667eea"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, y=col_sel, title=f"Box Plot — {col_sel}",
                         color_discrete_sequence=["#764ba2"])
            st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Correlation Heatmap")
        corr_matrix = df[numeric_cols].corr().round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

        pairs = top_correlations(df, threshold=0.5)
        if pairs:
            st.subheader("🏆 Top Correlated Pairs")
            st.dataframe(pd.DataFrame(pairs), hide_index=True, use_container_width=True)

    # Categorical analysis
    if cat_cols:
        st.subheader("📊 Categorical Analysis")
        cat_sel = st.selectbox("Select categorical column", cat_cols, key="cat_col")
        top_n = st.slider("Top N categories", 5, 20, 10, key="cat_topn")
        vc = df[cat_sel].value_counts().head(top_n).reset_index()
        vc.columns = [cat_sel, "Count"]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(vc, x=cat_sel, y="Count", title=f"Top {top_n}: {cat_sel}",
                         color="Count", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(vc, names=cat_sel, values="Count",
                         title=f"Share by {cat_sel}")
            st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    if len(numeric_cols) >= 2:
        st.subheader("🔵 Scatter Plot")
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, key="scatter_x")
        y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
        color_col = c3.selectbox("Color by", ["None"] + cat_cols, key="scatter_color")
        color = None if color_col == "None" else color_col
        fig = px.scatter(df, x=x_col, y=y_col, color=color,
                         title=f"{x_col} vs {y_col}",
                         opacity=0.7, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    if date_cols and numeric_cols:
        st.subheader("📅 Time Series")
        date_sel = st.selectbox("Date column", date_cols, key="ts_date")
        val_sel = st.selectbox("Value column", numeric_cols, key="ts_val")
        ts_df = df[[date_sel, val_sel]].dropna().sort_values(date_sel)
        fig = px.line(ts_df, x=date_sel, y=val_sel,
                      title=f"{val_sel} over time")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 4 — INSIGHTS
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">💡 Data Insights</div>', unsafe_allow_html=True)

    st.subheader("🔍 Automated Insights (Rule-Based)")
    for insight in st.session_state.local_insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("🤖 AI-Powered Insights (OpenAI)")

    if st.session_state.ai_insights_text:
        st.markdown(st.session_state.ai_insights_text)
    else:
        st.info("Click **💡 AI Insights** in the sidebar to generate AI analysis.")

    st.divider()
    st.subheader("📊 Dashboard Suggestions")

    if st.session_state.ai_dashboard_text:
        st.markdown(st.session_state.ai_dashboard_text)
    else:
        st.info("Click **📊 Dashboard Ideas** in the sidebar to generate suggestions.")


# ════════════════════════════════════════════════════════════
#  TAB 5 — ASK AI
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">❓ Ask Business Questions</div>', unsafe_allow_html=True)
    st.caption("Ask any question about your data in plain English. The AI will analyze and answer.")

    # Quick question buttons
    st.subheader("Quick Questions")
    quick_qs = [
        "What are the top 5 trends in this data?",
        "Which column has the most impact on the outcome?",
        "What is the average and distribution of the main metric?",
        "Are there any anomalies or unusual patterns?",
        "What time period does this data cover?",
        "Which category performs best?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        with cols[i % 3]:
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                st.session_state["current_question"] = q

    st.divider()
    user_question = st.text_area(
        "Your Question",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g. What is the total revenue by region? Which product sells best?",
        height=80,
        key="qa_input",
    )

    if st.button("🔍 Get Answer", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        elif not os.getenv("OPENAI_API_KEY", ""):
            st.error("❌ OpenAI API key required. Add it in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = ai_answer_question(df, user_question, model=ai_model)
                    st.session_state.qa_log.append({
                        "question": user_question,
                        "answer": answer,
                    })
                    st.session_state["current_question"] = ""
                except Exception as e:
                    st.error(f"❌ {e}")

    # Show Q&A history
    if st.session_state.qa_log:
        st.divider()
        st.subheader("💬 Conversation History")
        for i, item in enumerate(reversed(st.session_state.qa_log)):
            with st.expander(f"Q: {item['question']}", expanded=(i == 0)):
                st.markdown(item["answer"])

        if st.button("💾 Save Q&A Log"):
            path = save_qa_log(st.session_state.qa_log, st.session_state.filename)
            st.success(f"Saved to `{path}`")


# ════════════════════════════════════════════════════════════
#  TAB 6 — REPORT
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">📄 Full Analysis Report</div>', unsafe_allow_html=True)

    if not st.session_state.report_text:
        st.info("Click **📄 Generate Full Report** in the sidebar to create the report.")

        st.subheader("Report will include:")
        features = [
            "✅ Dataset overview and shape comparison",
            "✅ Data cleaning steps taken",
            "✅ Descriptive statistics",
            "✅ Automated insights",
            "✅ AI-powered analysis (if generated)",
            "✅ Dashboard suggestions (if generated)",
        ]
        for f in features:
            st.markdown(f)
    else:
        st.success(f"Report saved to: `{st.session_state.report_path}`")
        st.text_area("Report Preview", st.session_state.report_text, height=600)
        st.download_button(
            "⬇️ Download Full Report",
            data=st.session_state.report_text,
            file_name=f"report_{st.session_state.filename}.txt",
            mime="text/plain",
        )
