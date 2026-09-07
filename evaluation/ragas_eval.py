import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from ragas import evaluate

from ragas.embeddings import (
    LangchainEmbeddingsWrapper,
)

from ragas.llms import (
    LangchainLLMWrapper,
)

from ragas.metrics import (
    AnswerRelevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from ragas.run_config import RunConfig

from config.settings import settings

from evaluation.metrics_store import save_metrics

from rag.chain.rag_chain import get_rag_chain


METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


# Seconds between questions during RAG chain phase.
# Each question triggers one embed_query call for MMR retrieval.
EMBEDDING_DELAY_SECONDS = 20


# Cache file: saves RAG chain outputs so RAGAS can be retried
# without re-running the expensive generation phase.
CACHE_PATH = Path(__file__).parent / "eval_cache.json"


def _load_cache() -> Dict[str, Any] | None:

    """Load cached RAG chain outputs if they exist and are recent."""

    if not CACHE_PATH.exists():
        return None

    try:

        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except (json.JSONDecodeError, KeyError):

        return None


def _save_cache(data: Dict[str, Any]) -> None:

    """Save RAG chain outputs to cache file."""

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[INFO] RAG chain outputs cached to {CACHE_PATH}"
    )

    print(
        "[INFO] On retry, the RAG chain phase will be skipped."
    )


def run_ragas_evaluation(
    use_cache: bool = True,
) -> Dict[str, Any]:

    # ---------------------------------------------------------
    # 1. Load the golden dataset
    # ---------------------------------------------------------
    print(f"[VERIFY] Retriever: k={settings.retriever_k}, fetch_k={settings.retriever_fetch_k}, lambda_mult={settings.retriever_lambda_mult}")
    dataset_path = (
        Path(__file__).parent
        / "golden_dataset.json"
    )

    with open(
        dataset_path,
        "r",
        encoding="utf-8",
    ) as f:

        golden_data: List[Dict[str, Any]] = json.load(f)

    total = len(golden_data)

    # ---------------------------------------------------------
    # 2. RAG chain phase — skipped if valid cache exists
    # ---------------------------------------------------------

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    cache = (
        _load_cache()
        if use_cache
        else None
    )

    if (
        cache
        and cache.get("questions")
        and len(cache["questions"]) == total
    ):

        # Cache hit — load outputs from previous run

        print(
            "[INFO] Cache found. "
            "Skipping RAG chain phase."
        )

        print(
            f"[INFO] Loaded {total} answers "
            f"from {CACHE_PATH}\n"
        )

        questions = cache["questions"]
        answers = cache["answers"]
        contexts = cache["contexts"]
        ground_truths = cache["ground_truths"]

    else:

        # No cache — run the full RAG chain

        print(
            f"[INFO] Running {total} questions "
            "through NEXUS RAG chain."
        )

        print(
            f"[INFO] {EMBEDDING_DELAY_SECONDS}s delay "
            "between questions to respect embedding quota."
        )

        estimated = (
            total * EMBEDDING_DELAY_SECONDS
        ) // 60 + 1

        print(
            f"[INFO] Estimated RAG chain phase: "
            f"~{estimated} minutes.\n"
        )

        rag_chain = get_rag_chain()

        for idx, item in enumerate(
            golden_data,
            start=1,
        ):

            question = item["question"]

            ground_truth = item["ground_truth"]

            print(
                f"  [{idx}/{total}] "
                f"{question[:70]}..."
            )

            result = asyncio.run(
                rag_chain.ainvoke(
                    {
                        "input": question,
                        "chat_history": [],
                    }
                )
            )

            generated_answer: str = (
                result.get("answer", "")
            )

            retrieved_docs = (
                result.get("context", [])
            )

            context_chunks: List[str] = [
                doc.page_content
                for doc in retrieved_docs
            ]

            questions.append(question)

            answers.append(generated_answer)

            contexts.append(context_chunks)

            ground_truths.append(ground_truth)

            if idx < total:

                print(
                    f"  [WAIT] Sleeping "
                    f"{EMBEDDING_DELAY_SECONDS}s "
                    "before next question..."
                )

                time.sleep(
                    EMBEDDING_DELAY_SECONDS
                )

        print(
            f"\n[INFO] RAG chain complete. "
            f"{total} answers and contexts collected."
        )

        # Save outputs so the next run can skip this phase

        _save_cache(
            {
                "questions": questions,
                "answers": answers,
                "contexts": contexts,
                "ground_truths": ground_truths,
                "cached_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

    # ---------------------------------------------------------
    # 3. Build RAGAS evaluation dataset
    # ---------------------------------------------------------

    evaluation_dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
        }
    )

    # ---------------------------------------------------------
    # 4. Configure evaluator LLM and embeddings
    # ---------------------------------------------------------

    evaluator_llm = ChatGoogleGenerativeAI(
        model=settings.analyst_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )

    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

    wrapped_llm = LangchainLLMWrapper(
        evaluator_llm
    )

    wrapped_embeddings = (
        LangchainEmbeddingsWrapper(
            evaluator_embeddings
        )
    )

    # ---------------------------------------------------------
    # 5. RAGAS RunConfig
    #
    # timeout=600   → 10 minutes per job.
    #
    # answer_relevancy hits 429, waits max_wait=120s,
    # retries up to 15 times.
    #
    # max_retries=15 → enough retries to survive
    # sustained rate limiting.
    #
    # max_wait=120 → 2 minutes between retries.
    #
    # max_workers=1 → strictly sequential.
    # ---------------------------------------------------------

    run_config = RunConfig(
        timeout=600,
        max_retries=15,
        max_wait=120,
        max_workers=1,
    )

    # ---------------------------------------------------------
    # Configure answer_relevancy
    #
    # strictness=1 means RAGAS generates only one
    # synthetic question per answer.
    #
    # The default strictness is 3, which requires
    # multiple candidates from the evaluator model.
    #
    # Gemini 3.1 Flash-Lite does not support multiple
    # candidates, so strictness must be reduced to 1.
    # ---------------------------------------------------------

    answer_relevancy_metric = AnswerRelevancy(
        strictness=1
    )

    metrics = [
        faithfulness,
        answer_relevancy_metric,
        context_precision,
        context_recall,
    ]

    print(
        "[INFO] Starting RAGAS evaluation "
        "(sequential + extended timeout mode)."
    )

    print(
        "[INFO] answer_relevancy strictness=1 "
        "(single synthetic question)."
    )

    print(
        "[INFO] answer_relevancy makes embedding "
        "calls internally."
    )

    print(
        "[INFO] On 429: RAGAS waits 120s and retries "
        "(up to 15 times, within 600s per job)."
    )

    print(
        "[INFO] Expected total runtime: "
        "45-60 minutes. Do not interrupt.\n"
    )

    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
        run_config=run_config,
    )

    # ---------------------------------------------------------
    # 6. Extract scores from EvaluationResult
    # ---------------------------------------------------------

    results_dict: Dict[str, Any] = {}

    for metric_name in METRIC_NAMES:

        try:

            per_sample_scores: List[float] = (
                ragas_result[metric_name]
            )

            valid_scores = [
                s
                for s in per_sample_scores
                if isinstance(s, (int, float))
                and not math.isnan(s)
            ]

            failed_count = (
                len(per_sample_scores)
                - len(valid_scores)
            )

            results_dict[metric_name] = (
                sum(valid_scores)
                / len(valid_scores)
                if valid_scores
                else float("nan")
            )

            results_dict[
                f"{metric_name}_successful"
            ] = len(valid_scores)

            results_dict[
                f"{metric_name}_failed"
            ] = failed_count

        except (KeyError, TypeError) as e:

            print(
                f"[WARN] Could not extract "
                f"{metric_name}: {e}"
            )

            results_dict[metric_name] = (
                float("nan")
            )

            results_dict[
                f"{metric_name}_successful"
            ] = 0

            results_dict[
                f"{metric_name}_failed"
            ] = len(golden_data)

    results_dict["timestamp"] = (
        datetime.now(timezone.utc).isoformat()
    )

    results_dict["questions_evaluated"] = (
        len(golden_data)
    )

    # ---------------------------------------------------------
    # 7. Quality warnings
    # ---------------------------------------------------------

    for metric_name in METRIC_NAMES:

        failed = results_dict.get(
            f"{metric_name}_failed",
            0,
        )

        if failed > len(golden_data) // 3:

            print(
                f"[WARN] {metric_name}: "
                f"{failed}/{len(golden_data)} jobs failed. "
                f"Scores may be unreliable. "
                f"Check API rate limits."
            )

    # ---------------------------------------------------------
    # 8. Persist and return
    # ---------------------------------------------------------

    save_metrics(results_dict)

    return results_dict