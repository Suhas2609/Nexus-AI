import streamlit as st
from components.citation_card import render_sources

def render_chat_history(messages: list[dict]):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    render_sources(msg["sources"])
            
            if "agent_trace" in msg and msg["agent_trace"] is not None:
                with st.expander("View Agent Reasoning Trace"):
                    st.text(msg["agent_trace"])