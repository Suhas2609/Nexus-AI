import streamlit as st
import requests
import pandas as pd

from config.settings import settings

# Phase 1: Page Configuration
st.set_page_config(page_title="Evaluation Dashboard", layout="wide")
st.title("System Evaluation & Metrics")

# Phase 2: Data Retrieval
try:
    response = requests.get(f"{settings.fastapi_base_url}/api/v1/metrics", timeout=5.0)
    if response.status_code == 200:
        json_data = response.json()
        latest = json_data.get("latest")
        history = json_data.get("history", [])
    else:
        latest = None
        history = []
        st.error(f"Failed to fetch metrics: HTTP {response.status_code}")
except requests.exceptions.RequestException as e:
    latest = None
    history = []
    st.error(f"API Connection Error: {str(e)}")

# Phase 3: Manual Execution Trigger
if st.button("Run Pipeline Evaluation"):
    with st.spinner("Executing RAGAS evaluation... (This takes 2-5 minutes)"):
        try:
            eval_response = requests.post(f"{settings.fastapi_base_url}/api/v1/evaluate")
            if eval_response.status_code == 200:
                st.success("Evaluation pipeline completed successfully.")
                st.rerun()
            else:
                st.error(f"Evaluation failed: HTTP {eval_response.status_code} - {eval_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: {str(e)}")

# Phase 4: Latest Metrics Visualization
if latest:
    st.subheader(f"Latest Evaluation: {latest.get('timestamp', 'Unknown Time')}")
    cols = st.columns(4)
    
    faithfulness = latest.get("faithfulness", 0.0) * 100
    answer_relevancy = latest.get("answer_relevancy", 0.0) * 100
    context_precision = latest.get("context_precision", 0.0) * 100
    context_recall = latest.get("context_recall", 0.0) * 100
    
    cols[0].metric(label="Faithfulness", value=f"{faithfulness:.1f}%")
    cols[1].metric(label="Answer Relevancy", value=f"{answer_relevancy:.1f}%")
    cols[2].metric(label="Context Precision", value=f"{context_precision:.1f}%")
    cols[3].metric(label="Context Recall", value=f"{context_recall:.1f}%")

# Phase 5: Historical Data Charting
if history:
    st.subheader("Historical Performance")
    df = pd.DataFrame(history)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
    metrics_cols = [col for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if col in df.columns]
    
    if metrics_cols:
        metrics_df = df[metrics_cols] * 100
        st.line_chart(metrics_df)  