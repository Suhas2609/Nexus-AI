import streamlit as st
import requests
import uuid

from config.settings import settings
from frontend.components.chat_ui import render_chat_history

# Phase 1: Page Configuration
st.set_page_config(page_title="Agent Mode", page_icon="🤖")
st.title("Multi-Agent Reasoning")
st.markdown("⚠️ *Note: Agent execution utilizes a multi-step analytical loop. Responses require 30-60 seconds to process.*")

# Phase 2: State Initialization (Memory Isolation)
if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = str(uuid.uuid4())

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# Phase 3: Historical Rendering
render_chat_history(st.session_state.agent_messages)

# Phase 4: User Input Capture
if prompt := st.chat_input("Ask a complex question requiring deep analysis..."):
    
    # Phase 5: Request Execution & State Update
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.spinner("Agents are researching and analyzing (this may take 30-60 seconds)..."):
        payload = {
            "session_id": st.session_state.agent_session_id,
            "query": prompt,
            "use_agent": True
        }
        
        try:
            response = requests.post(f"{settings.fastapi_base_url}/api/v1/chat", json=payload)
            
            if response.status_code == 200:
                json_data = response.json()
                st.session_state.agent_messages.append({
                    "role": "assistant",
                    "content": json_data["answer"],
                    "sources": json_data.get("sources", []),
                    "agent_trace": json_data.get("agent_trace")
                })
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: {str(e)}")
            
    st.rerun()