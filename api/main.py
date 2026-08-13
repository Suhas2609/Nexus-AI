import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Core Configurations & Singletons ---
from tracing.langsmith_config import configure_tracing
from rag.vectorstore.store import get_or_create_store
from rag.chain.rag_chain import get_rag_chain
from agents.graph import get_agent_graph

# --- API Routers ---
from api.routes.health import router as health_router
from api.routes.documents import router as documents_router
from api.routes.chat import router as chat_router
from api.routes.evaluation import router as evaluation_router

# Configure logging format for standard tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Phase 1 & 2: Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle. Pre-loads heavy vector resources, 
    compiles analytical graph structures, and ensures tracing setup 
    precedes any execution logic.
    """
    logger.info("Initializing NEXUS AI application services...")
    
    try:
        # Step 1: Initialize tracing configuration before object creation
        configure_tracing()
        
        # Step 2: Establish connection and warm the Chroma client instance
        store = get_or_create_store()
        
        # Step 3: Build, configure, and cache the LCEL execution chain
        get_rag_chain()
        
        # Step 4: Compile and cache the LangGraph state machine workflow
        get_agent_graph()
        
        logger.info("All core engines and state graphs compiled successfully.")
        
        # Step 5: Yield references to application state storage dict
        yield {"chroma_store": store}
        
    except Exception as e:
        logger.critical(f"Fatal exception during app lifespan initialization: {e}", exc_info=True)
        raise e
        
    finally:
        # Phase 2: App Teardown / Cleanup
        logger.info("NEXUS AI API shutting down cleanly...")


# Phase 3: Application Factory Initialization
app = FastAPI(
    title="NEXUS AI",
    version="1.0.0",
    lifespan=lifespan
)


# Phase 4: Middleware Configuration
# Fixed: Removed allow_credentials=True to prevent fatal startup AssertionError with wildcard origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Phase 5: Router Registration
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["Evaluation"])