import pandas as pd
import streamlit as st

def render_historical_chart(history_data: list) -> None:
    """Parses evaluation history and renders a line chart of the metrics over time."""
    if not history_data:
        st.info("No historical evaluation data available.")
        return

    try:
        # Convert dictionary list to DataFrame
        df = pd.DataFrame(history_data)
        
        # Ensure timestamp is a datetime object and set it as the index for time-series plotting
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        
        # Filter for numeric columns (the RAGAS metrics)
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        
        if not numeric_cols.empty:
            st.line_chart(df[numeric_cols])
        else:
            st.warning("No numeric metric data available to plot.")
            
    except Exception as e:
        st.error(f"Failed to render chart: {e}")