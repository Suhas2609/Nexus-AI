import streamlit as st

def render_sources(sources: list) -> None:
    """Renders a list of source documents as styled markdown blocks."""
    if not sources:
        st.write("No sources provided.")
        return
        
    for i, source in enumerate(sources, start=1):
        filename = source.get("source", "Unknown Document")
        page = source.get("page", "N/A")
        
        # Styled HTML block for the citation
        card_html = f"""
        <div style="
            border-left: 4px solid #4CAF50; 
            background-color: rgba(76, 175, 80, 0.1); 
            padding: 10px 15px; 
            margin-bottom: 10px; 
            border-radius: 4px;">
            <p style="margin: 0; font-weight: bold; font-size: 14px;">Source {i}: {filename}</p>
            <p style="margin: 0; font-size: 12px; color: gray;">Page: {page}</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)