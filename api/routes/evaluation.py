import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

# --- Schemas ---
from api.models.schemas import EvaluationResponse, MetricsResponse

# --- Evaluation Engine & Storage ---
from evaluation.ragas_eval import run_ragas_evaluation
from evaluation.metrics_store import load_metrics_history

logger = logging.getLogger(__name__)

# Phase 1: Router Initialization
router = APIRouter()


# Phase 2: Trigger Evaluation Endpoint (POST /evaluate)
@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint():
    """
    Triggers the RAGAS offline evaluation suite against the golden dataset.
    
    NOTE: This is an intensive offline batch process that utilizes LLM-as-a-judge
    metrics. Expect an execution latency of 2 to 5 minutes depending on the 
    size of the golden dataset. This route blocks execution during processing.
    """
    # Step 1: Pre-flight Validation
    project_root = Path(__file__).resolve().parents[2]
    golden_dataset_path = project_root / "evaluation" / "golden_dataset.json"

    if not golden_dataset_path.exists():
        logger.warning(f"Evaluation aborted: Golden dataset missing at {golden_dataset_path}")
        raise HTTPException(
            status_code=404, 
            detail="Golden dataset not found. Run scripts/seed_golden_dataset.py first."
        )

    try:
        # Step 2: Execution
        logger.info("Initiating offline batch RAGAS evaluation pipeline...")
        evaluation_results = await run_ragas_evaluation()
        logger.info("RAGAS metrics calculation complete and logged to historical store.")

        # Step 3: Response Assembly
        return EvaluationResponse(
            faithfulness=evaluation_results.get("faithfulness", 0.0),
            answer_relevancy=evaluation_results.get("answer_relevancy", 0.0),
            context_precision=evaluation_results.get("context_precision", 0.0),
            context_recall=evaluation_results.get("context_recall", 0.0),
            timestamp=evaluation_results.get("timestamp", ""),
            questions_evaluated=evaluation_results.get("questions_evaluated", 0)
        )

    except Exception as e:
        logger.error(f"RAGAS evaluation execution failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during evaluation execution: {str(e)}"
        )


# Phase 3: Fetch Metrics Endpoint (GET /metrics)
@router.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """
    Retrieves historical metrics logs for evaluation trend tracking.
    Safely handles uninitialized files by returning empty payloads.
    """
    try:
        # Step 1: Data Extraction
        history = load_metrics_history()

        # Step 2: State Calculation
        if history and len(history) > 0:
            latest = history[-1]
        else:
            latest = None

        # Step 3: Response Delivery
        return MetricsResponse(
            history=history,
            latest=latest
        )

    except Exception as e:
        logger.error(f"Failed to compile metrics response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error parsing historical evaluation logs."
        )