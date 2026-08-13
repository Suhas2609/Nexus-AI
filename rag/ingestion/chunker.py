import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings


def chunk_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    
    for chunk in chunks:
        content_bytes = chunk.page_content.encode("utf-8")
        chunk_id = hashlib.md5(content_bytes).hexdigest()
        chunk.metadata["chunk_id"] = chunk_id
    
    return chunks
 