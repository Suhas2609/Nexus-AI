import streamlit as st
import requests
from config.settings import settings

#Phase 1: Global Page Configuration
st.set_page_config(page_title = "NEXUS-AI", page_icon = "🔍", layout="wide")


#Phase 2: Main Body Scaffolding
st.title("NEXUS AI")
st.markdown("Enterprise-grade Retrieval-Augmented Generation and Multi-Agent Reasoning System.")

#Phase 3: Sidebar Status Dashboard
st.sidebar.subheader("System Status")
st.sidebar.code(settings.fastapi_base_url, language="text")

try:
    response = requests.get(f"{settings.fastapi_base_url}/api/v1/health", timeout=2.0)
    if response.status_code == 200 and response.json().get("status") == "ok":
        st.sidebar.success("🟢 API Online")
    else:
        st.sidebar.error("🔴 API Offline")
    
except requests.RequestException:
    st.sidebar.error("🔴 API Offline")