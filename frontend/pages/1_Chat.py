import streamlit as st
import requests
import uuid
from config.settings import settings
from frontend.components.chat_ui import render_chat_history

#Phase 1: Page Configuration
st.set_page_config(page_title="Simple Chat", page_icon="💬")
st.title("Document Chat")

#Phase 2: State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

#Phase 3: Historical Rendering
render_chat_history(st.session_state.messages)

#Phase 4: User Input Capture
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Phase 5: Request Execution & State Update
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.spinner("Retrieving answers..."):
        payload = {
            "session_id": st.session_state.session_id,
            "query": prompt,
            "use_agent": False
        }
        
        try:
            response = requests.post(f"{settings.fastapi_base_url}/api/v1/chat", json=payload)
            
            if response.status_code == 200:
                json_data = response.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": json_data["answer"],
                    "sources": json_data.get("sources", [])
                })
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: {str(e)}")
            
    st.rerun()