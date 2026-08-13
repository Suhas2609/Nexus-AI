import os
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

def configure_tracing() -> None:
    """
    Injects LangSmith configuration into the environment for automatic telemetry.
    Must be called before any LangChain objects are instantiated.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        logger.info(f"LangSmith tracing enabled for project: {settings.langchain_project}")
    else:
        logger.warning("LANGCHAIN_API_KEY is empty. Tracing is disabled.")