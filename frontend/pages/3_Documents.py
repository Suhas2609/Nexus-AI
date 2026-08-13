import streamlit as st
import requests
import pandas as pd

from config.settings import settings

# Phase 1: Page Configuration
st.set_page_config(page_title="Document Management", page_icon="📂")
st.title("Knowledge Base Management")

# Phase 2: Ingestion Interface
st.subheader("Upload New Document")
uploaded_file = st.file_uploader("Select a document to add to the knowledge base", type=["pdf", "txt", "md"])

if st.button("Ingest Document"):
    if uploaded_file is not None:
        with st.spinner("Processing and vectorizing..."):
            # Construct the binary file payload matching FastAPI's UploadFile requirement
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post(f"{settings.fastapi_base_url}/api/v1/upload", files=files)
                
                if response.status_code == 200:
                    st.success(f"Successfully vectorized and stored: {uploaded_file.name}")
                else:
                    st.error(f"Upload Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"API Connection Error: {str(e)}")
    else:
        st.warning("Please select a file before attempting ingestion.")

# Phase 3: Corpus Auditing Divider
st.divider()
st.subheader("Current Corpus")

# Phase 4 & 5: State Retrieval and Data Visualization
try:
    response = requests.get(f"{settings.fastapi_base_url}/api/v1/documents")
    
    if response.status_code == 200:
        json_data = response.json()
        total_chunks = json_data.get("total_chunks", 0)
        documents_list = json_data.get("documents", [])
        
        # Step 1: Metrics Summary
        st.metric(label="Total Vector Chunks", value=total_chunks)
        
        # Step 2: Table Rendering
        if not documents_list:
            st.info("No documents currently in the vector store.")
        else:
            df = pd.DataFrame(documents_list)
            # Optional cleanup to make the table columns look cleaner
            if "source" in df.columns:
                df = df.rename(columns={"source": "Filename", "estimated_pages": "Estimated Pages", "chunk_count": "Chunk Count"})
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.error(f"Failed to fetch documents. Status Code: {response.status_code}")
        
except requests.exceptions.RequestException as e:
    st.error(f"API Connection Error: {str(e)}")