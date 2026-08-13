from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document

def load_documents(path: str) ->  List[Document]:
    """
    Loads documents based on path type: Directory (PDFs), Single PDF, or Text/Markdown.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Error: The path '{path}' does not exist.")
        
    if file_path.is_dir():
        loader = DirectoryLoader(
            path,
            glob = "**/*.pdf",
            loader_cls = PyPDFLoader
        )
        docs = loader.load()
    
    elif file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()

    elif file_path.suffix.lower() in  [".txt", ".md"]:
        loader = TextLoader(path)
        docs = loader.load()

    else:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. "
            "Please provide a PDF, a directory of PDFs, or a Text/Markdown file."
        )
    

    for doc in docs:
        full_source = doc.metadata.get("source", "")
        doc.metadata["source"] = Path(full_source).name

        page_val = doc.metadata.get("page", 0)
        doc.metadata["page"] = int(page_val)

    return docs

    