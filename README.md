# Nexus AI

Multi-agent research platform with RAG, LangGraph orchestration, FastAPI backend, and Streamlit frontend.

## Phase 0 — Project Foundation

This repository is being built in phases. Phase 0 includes project configuration and structure only.

### Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

4. Verify configuration:

```bash
python -c "from config.settings import settings; print(settings.analyst_model)"
```

Expected output: `gemini-1.5-flash`

### Environment Variables

See `.env.example` for all required variables.

### License

MIT
