# NEXUS-AI

NEXUS-AI is an enterprise-grade document research platform built with a multi-agent RAG (Retrieval-Augmented Generation) architecture, FastAPI backend, Streamlit frontend, and external n8n automation workflows.

---

## Architecture Overview

```
Documents
    ↓
Ingestion / Loading (PyPDFLoader / DirectoryLoader)
    ↓
Recursive Chunking (1000 chars, 200 overlap)
    ↓
Gemini Embeddings (gemini-embedding-001)
    ↓
Chroma Vector Store (Persistent)
    ↓
LangGraph Multi-Agent Engine
    ├── Orchestrator (Intent Classifier / Router)
    ├── Retriever (MMR Search)
    ├── [Optional] Web Search (DuckDuckGo fallback)
    ├── Analyst (Evidence Synthesis)
    ├── Critic (QA Auditor & Fact Verifier)
    │     ├── APPROVE ───────────┐
    │     └── REVISE (Max 2) ──→ Analyst
    ↓
Report Writer (Final Synthesizer)
    ↓
Final Answer & Sources
```

### Multi-Agent LangGraph Workflow
1. **Orchestrator Node**: Evaluates user query to determine whether standard document retrieval is sufficient or web search fallback is required.
2. **Retriever Node**: Executes MMR vector search against ChromaDB.
3. **Web Search Node** *(Optional)*: Triggers external web search if domain knowledge extends beyond ingested documents.
4. **Analyst Node**: Synthesizes retrieved evidence into an initial factual response.
5. **Critic Node**: Performs strict QA audit against ground-truth evidence. Evaluates completeness, factual grounding, and missing requirements. Outputs a structured verdict (`APPROVE` or `REVISE`). If `REVISE` is issued, feedback routes back to the Analyst for refinement. The revision loop is strictly capped at **2 iterations**. If the Critic output is unparseable or missing a verdict, it defaults safely to `REVISE`.
6. **Report Writer Node**: Formats approved responses into final user-facing Markdown with inline source attributions.

---

## Retrieval Configuration

- **Vector Database**: ChromaDB (Persistent storage in `./data/chroma`)
- **Search Algorithm**: Maximal Marginal Relevance (MMR)
- **Parameters**:
  - `k` = `4` (Final retrieved chunks per query)
  - `fetch_k` = `20` (Candidate pool size before MMR filtering)
  - `lambda_mult` = `0.70` (Diversity vs. relevance trade-off)
- **Embedding Model**: `models/gemini-embedding-001`
  - Uses `task_type="retrieval_document"` during document chunk ingestion.
  - Uses `task_type="retrieval_query"` during user query search execution.

---

## Offline RAGAS Evaluation

NEXUS-AI incorporates an offline batch evaluation pipeline powered by RAGAS and LLM-as-a-Judge metrics.

### Evaluation Metrics
- **Faithfulness**: Verifies that generated statements are derived exclusively from retrieved context.
- **Answer Relevancy**: Assesses how directly the answer addresses the user prompt.
- **Context Precision**: Evaluates whether relevant chunks are ranked higher than irrelevant ones.
- **Context Recall**: Verifies that ground-truth answers are present in retrieved contexts.

### Execution & Storage
Evaluation is **intentionally decoupled** from real-time query serving. It runs as an offline batch process against `evaluation/golden_dataset.json`.
- Results are saved to `data/evaluation/metrics_history.json`.
- Historical evaluation metrics are exposed via `GET /api/v1/metrics`.

> **Note**: RAGAS evaluation is NOT executed on live user queries.

---

## External n8n Automation Layer

n8n operates as an **external automation orchestrator** around NEXUS-AI and is NOT part of the synchronous RAG request pipeline.

```
Core NEXUS-AI Service:
User / Frontend → FastAPI → LangGraph Agent → ChromaDB → Answer

External Automation Layer (n8n):
n8n Triggers → Scheduled Ingestion / Briefings / Quality Monitoring
```

### Workflows Exposed
1. **Document Ingestion (`document_ingestion.json`)**: Automatically scans `/data/documents/` and invokes `POST /api/v1/upload` when new files are dropped.
2. **Daily Digest (`daily_digest.json`)**: Runs on a scheduled trigger to poll key metrics or query summary reports via `POST /api/v1/chat`.
3. **Quality Alert (`quality_alert.json`)**: Runs on a scheduled trigger to inspect system health:
   - Polls `GET http://nexus_api:8000/api/v1/metrics`
   - Evaluates `latest.faithfulness < 0.75`
   - Dispatches a webhook alert if quality degradation is detected.
   - *Note*: Does NOT execute `POST /api/v1/evaluate`.

---

## Production & Engineering Architecture

- **Backend**: FastAPI (Async framework with Pydantic v2 schemas and CORS middleware).
- **Frontend**: Streamlit (Interactive chat interface with agent tracing toggle).
- **Containerization**: Docker Compose (`nexus_api`, `nexus_frontend`, `n8n` services with isolated bridge networks).
- **Testing**: `pytest` and `pytest-asyncio` unit/integration test suite.
- **Environment Management**: Centralized configuration via `pydantic-settings` reading `.env`.

---

## Known Limitation & Production Roadmap

### Known Limitation: Query-Dilution on Complex Multi-Part Queries
When executing complex, multi-part questions spanning multiple distinct topics or documents, a single dense query embedding can dilute specific retrieval intents.

#### Diagnostic Findings
- For broad multi-part queries, specific quantitative evidence (e.g., parameter counts on deep pages) ranked as low as **rank 35** in initial vector search, falling outside standard `fetch_k=20` / `k=4` retrieval bounds.
- When decomposed into targeted subqueries, the exact same evidence consistently retrieved at **rank 1**.
- Simply increasing `fetch_k` or `k` was deemed unsuitable for production due to context window pollution.

#### Production Roadmap
To address query-dilution, the following architectural enhancement is planned:
1. **LLM Sub-Query Decomposition**: Deconstruct complex user prompts into discrete sub-queries.
2. **Per-Intent Parallel Retrieval**: Execute independent MMR vector searches per sub-query.
3. **Rank-Deduplication & Budgeting**: Merge and deduplicate candidate contexts within a strict evidence token budget.

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/health` | `GET` | System liveness, active LLM model, Chroma collection count, and tracing status. |
| `/api/v1/chat` | `POST` | Primary query route. Accepts `session_id`, `query`, and `use_agent` boolean flag. |
| `/api/v1/upload` | `POST` | Uploads PDF/TXT/MD files and triggers the chunking/ingestion pipeline. |
| `/api/v1/documents` | `GET` | Returns list of ingested documents, page counts, and total chunks in vector store. |
| `/api/v1/evaluate` | `POST` | Triggers offline batch RAGAS evaluation against `golden_dataset.json`. |
| `/api/v1/metrics` | `GET` | Returns full evaluation history and `latest` score breakdown. |

---

## Setup & Local Execution

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Google Gemini API Key (`GEMINI_API_KEY`)
- Groq API Key (`GROQ_API_KEY`)

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
ANALYST_MODEL=gemini-1.5-flash
CRITIC_MODEL=llama-3.3-70b-versatile
ORCHESTRATOR_MODEL=gemini-1.5-flash
REPORT_WRITER_MODEL=gemini-1.5-flash
n8n_password=admin_password
```

### 3. Running with Docker Compose
```bash
docker compose -f docker/docker-compose.yml up --build -d
```
- **FastAPI Backend**: `http://localhost:8000/docs`
- **Streamlit Interface**: `http://localhost:8501`
- **n8n Automation**: `http://localhost:5678`

### 4. Running Tests
```bash
# Execute non-RAGAS unit/integration tests
docker exec nexus_api pytest tests/ -k "not ragas and not eval" -v
```

---

## License

MIT License
