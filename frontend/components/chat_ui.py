import streamlit as st

def render_chat_history(messages: list[dict]):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources"):
                    for source in msg["sources"]:
                        st.markdown(f"- {source.get('source', 'Unknown')} (Page {source.get('page', 'Unknown')})")
            
            if "agent_trace" in msg and msg["agent_trace"] is not None:
                with st.expander("🧠 View Agent Reasoning Trace"):
                    st.text(msg["agent_trace"])