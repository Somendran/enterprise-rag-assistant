# Enterprise RAG Assistant

Production-oriented Retrieval-Augmented Generation (RAG) system for enterprise documents.

Stack:
- Backend: FastAPI + LangChain + FAISS
- Retrieval: Hybrid vector + BM25 with optional neural reranking
- Generation: OpenAI (optional primary) with local Ollama fallback
- Frontend: React + TypeScript + Vite

## What it does

- Ingest one or more PDF files into a local FAISS-backed knowledge base.
- Answer questions with grounded responses and explicit source references.
- Stream answers in real time over Server-Sent Events (SSE).
- Provide confidence scoring and diagnostics for retrieval and generation quality.
- Avoid duplicate indexing using content hashes.

## Pipeline (Current)

1. Retrieval: vector and BM25 branches run in parallel and are fused.
2. Rerank: BGE cross-encoder reranker reorders top candidates (with skip/gating rules).
3. Generation: OpenAI or local Ollama path builds final answer.
4. Verification layer: lightweight claim/citation checks run post-generation.
5. Output: response includes answer, sources, confidence, and diagnostics.

## Key features

- Overlapping chunking for better context continuity.
- Layout-aware PDF extraction with table rendering and section hints.
- Hybrid retrieval with weighted score fusion.
- Adaptive fast mode for short/simple queries.
- Multi-stage retrieval with reranker scope caps and skip heuristics.
- Deterministic low-confidence retrieval fallback.
- Query cache keyed by normalized query, selected chunks, and prompt fingerprint.
- Post-generation verification diagnostics:
  - claim support counts
  - citation coverage
  - invalid citation list
  - verification timing and failure flags

## Project layout

- `backend/`: FastAPI API, ingestion, retrieval, generation, verification services
- `frontend/`: chat UI with streaming response rendering
- `data/`: uploads and persisted FAISS index artifacts

## Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama installed and running locally (`http://localhost:11434`) if using local generation
- OpenAI API key if enabling OpenAI path

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env` from `backend/.env.example`, then set your runtime values.

If using local Ollama, pull your configured model (default in code: `gemma4:e2b`):

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull gemma4:e2b
```

Run backend:

```bash
uvicorn app.main:app --reload --app-dir backend
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Default backend target is `http://localhost:8000`.

## API overview

### `POST /upload`

Batch upload endpoint for one or more PDFs.

Response includes per-file status and aggregate counts:
- `files[]` (`success`, `duplicate`, or `failed`)
- `total_files`
- `processed_files`
- `total_chunks_indexed`

### `POST /knowledge-base/reset`

Clears uploaded files, FAISS index artifacts, and query cache.

### `POST /query`

Returns full structured response:
- `answer`
- `sources`
- `confidence_score`
- `confidence_level`
- `diagnostics` (when enabled)

### `POST /query/stream`

Streams partial answer chunks over SSE (`chunk`, `done`, `error` events), then returns final payload metadata in `done`.

### `GET /health`

Service health check.

## Important configuration

See `backend/app/config.py` for the full authoritative list.

Core runtime controls:
- LLM routing and budgets: `USE_OPENAI`, `OPENAI_*`, `LOCAL_LLM_*`, `LLM_MAX_TOKENS`
- Retrieval/rerank: `RETRIEVAL_*`, `BM25_*`, `VECTOR_WEIGHT`, `LEXICAL_WEIGHT`, `ENABLE_NEURAL_RERANKER`, `RERANK_*`
- Context shaping: `MAX_CONTEXT_CHARACTERS`, `FAST_MODE_*`, `CONTEXT_DOMINANT_GAP_THRESHOLD`
- Diagnostics/cache: `ENABLE_RETRIEVAL_DIAGNOSTICS`, `ENABLE_QUERY_CACHE`, `QUERY_CACHE_TTL_SECONDS`
- Verification layer: `ENABLE_VERIFICATION`, `VERIFICATION_SIMILARITY_THRESHOLD`, `VERIFICATION_MIN_ANSWER_CHARS`, `VERIFICATION_WARNING_SUPPORT_THRESHOLD`

## Limitations

- Image-only/scanned PDF content is not fully understood without OCR/vision augmentation.
- FAISS storage is local and best suited for single-instance deployment unless externalized.

## License

No repository license file is currently included.