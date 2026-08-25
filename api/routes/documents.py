import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from rag.ingestion.pipeline import ingest
from api.models.schemas import UploadResponse, DocumentsResponse, DocumentInfo
from rag.vectorstore.store import get_or_create_store

router = APIRouter()

UPLOAD_DIR = Path("./data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}"
        )
        
    target_path = UPLOAD_DIR / file.filename
    try:
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file to disk: {e}")
    finally:
        file.file.close()

    try:
        ingest_result = ingest(str(target_path))
        
        chunks_added = ingest_result.get("chunks_added", 0)
        chunks_skipped = ingest_result.get("chunks_skipped", 0) 
        total_in_store = ingest_result.get("total_in_store", 0)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {e}")

    return UploadResponse(
        source=file.filename,
        chunks_added=chunks_added,
        chunks_skipped=chunks_skipped,
        total_in_store=total_in_store,
        message=f"Successfully ingested {file.filename}"
    )

@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(request: Request) -> DocumentsResponse:
    try:
        store = get_or_create_store()
        
        collection_data = store.get(include=["metadatas"])
        metadatas = collection_data.get("metadatas", [])
        total_chunks = len(metadatas)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query vector database: {e}")

    if not metadatas:
        return DocumentsResponse(documents=[], total_chunks=0)

    doc_groups = {}
    for meta in metadatas:
        source_path = meta.get("source", "Unknown")
        source_name = Path(source_path).name 
        
        page = meta.get("page", 1)
        
        if source_name not in doc_groups:
            doc_groups[source_name] = page
        else:
            if page > doc_groups[source_name]:
                doc_groups[source_name] = page

    doc_info_list = [
        DocumentInfo(source=name, page_count=max_page)
        for name, max_page in doc_groups.items()
    ]
    
    return DocumentsResponse(
        documents=doc_info_list,
        total_chunks=total_chunks
    )