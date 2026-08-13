import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

# Upstream dependencies
from rag.ingestion.pipeline import ingest
from api.models.schemas import UploadResponse, DocumentsResponse, DocumentInfo

# Phase 1: Router Initialization
router = APIRouter()

# Ensure the upload directory exists locally
UPLOAD_DIR = Path("./data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Phase 2: File Upload Endpoint
@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document to the server and triggers the RAG ingestion pipeline."""
    
    # Step 1: Validation
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}"
        )
        
    # Step 2: Disk I/O (Safe Streaming)
    target_path = UPLOAD_DIR / file.filename
    try:
        # Open in binary write mode and stream the upload to disk
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file to disk: {e}")
    finally:
        # Explicitly close the uploaded file object
        file.file.close()

    # Step 3: Ingestion
    try:
        # Trigger the pipeline and capture the metrics dictionary
        ingest_result = ingest(str(target_path))
        
        chunks_added = ingest_result.get("chunks_added", 0)
        chunks_skipped = ingest_result.get("chunks_skipped", 0) 
        total_in_store = ingest_result.get("total_chunks", 0)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {e}")

    # Step 4: Response Assembly
    return UploadResponse(
        source=file.filename,
        chunks_added=chunks_added,
        chunks_skipped=chunks_skipped,
        total_in_store=total_in_store,
        message=f"Successfully ingested {file.filename}"
    )

# Phase 3: Corpus Listing Endpoint
@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(request: Request) -> DocumentsResponse:
    """Retrieves a summary of all documents currently indexed in the vector database."""
    
    # Step 1: Store Query
    try:
        store = getattr(request.app.state, "chroma_store", None)
        if not store:
            # Graceful fallback if store isn't attached yet
            return DocumentsResponse(documents=[], total_chunks=0)
            
        # Fetch metadata for all chunks in the store
        collection_data = store.get(include=["metadatas"])
        metadatas = collection_data.get("metadatas", [])
        total_chunks = len(metadatas)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query vector database: {e}")

    # Step 2: Grouping & Estimation
    if not metadatas:
        return DocumentsResponse(documents=[], total_chunks=0)

    doc_groups = {}
    for meta in metadatas:
        source_path = meta.get("source", "Unknown")
        # Extract just the filename from the absolute path
        source_name = Path(source_path).name 
        
        page = meta.get("page", 1)
        
        if source_name not in doc_groups:
            doc_groups[source_name] = page
        else:
            # Keep the highest page number found as an estimate of document length
            if page > doc_groups[source_name]:
                doc_groups[source_name] = page

    # Step 3: Response Assembly
    doc_info_list = [
        DocumentInfo(source=name, page_count=max_page)
        for name, max_page in doc_groups.items()
    ]
    
    return DocumentsResponse(
        documents=doc_info_list,
        total_chunks=total_chunks
    )