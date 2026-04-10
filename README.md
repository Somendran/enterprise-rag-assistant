# Enterprise RAG Assistant

FastAPI + React application for querying internal PDFs with a retrieval-augmented generation pipeline.

## What it does

- Upload PDFs into a FAISS-backed knowledge base.
- Ask natural-language questions against indexed documents.
- Return grounded answers with source citations.
- Expose confidence scores and retrieval diagnostics for quality tuning.
- Prevent duplicate uploads by hashing document content.

## Current architecture

- Backend: FastAPI, LangChain, FAISS, local Ollama generation (Gemma 4 by default).
- Frontend: React 19 + TypeScript + Vite.
- Embeddings: Local sentence-transformers (Hugging Face).
- LLM strategy: Local-first; optional Gemini fallback (disabled by default).
- Storage: Local uploads directory plus persisted FAISS index.

## Key features

- PDF ingestion with page extraction and overlapping chunking.
- Multi-query retrieval with rerank-lite scoring.
- One-pass deterministic retrieval fallback for low-confidence questions.
- Grounded answer generation with cleaned citations.
- Source metadata in responses, including relevance score.
- Duplicate upload detection using file hashes.
- In-process FAISS write locking for safer single-instance uploads.

## Repository layout

- `backend/` FastAPI app and RAG services.
- `frontend/` React chat UI.
- `data/` Local uploads and FAISS artifacts.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama installed and running locally (`http://localhost:11434`)
- Local Ollama model pulled (default: `gemma4:e4b`)
- Optional: Google API key only if you enable Gemini fallback

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a backend `.env` file from `backend/.env.example` and set at least:

- `LOCAL_LLM_ENDPOINT`
- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_VALIDATE_MODEL`

Pull the default local model before running queries:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull gemma4:e4b
```

Validate it exists:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" show gemma4:e4b
```

Run the API:

```bash
uvicorn app.main:app --reload --app-dir backend
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:8000` by default.

## API overview

### `POST /upload`

Uploads a PDF, splits it into chunks, embeds the content, and stores it in FAISS.

Response includes:

- `filename`
- `chunks_indexed`
- `message`

### `POST /query`

Accepts a question and returns:

- `answer`
- `sources`
- `confidence_score`
- `confidence_level`
- `diagnostics` when enabled

### `GET /health`

Basic health check for the API.

## Environment variables

Backend values are documented in `backend/.env.example`. Important ones include:

- `LOCAL_LLM_ENDPOINT`
- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_VALIDATE_MODEL`
- `LOCAL_LLM_TEMPERATURE`
- `LOCAL_LLM_NUM_PREDICT`
- `LOCAL_LLM_RETRY_NUM_PREDICT`
- `LOCAL_LLM_STREAM`
- `LOCAL_LLM_CONNECT_TIMEOUT_SECONDS`
- `LOCAL_LLM_READ_TIMEOUT_SECONDS`
- `LOCAL_LLM_NUM_GPU`
- `ENABLE_GEMINI_FALLBACK`
- `MAX_CONTEXT_CHARACTERS`

- `RETRIEVAL_TOP_K`
- `RETRIEVAL_CANDIDATE_K`
- `RETRIEVAL_LOW_CONFIDENCE_THRESHOLD`
- `ANSWER_LOW_CONFIDENCE_THRESHOLD`
- `ENABLE_RETRIEVAL_FALLBACK`
- `ENABLE_RETRIEVAL_DIAGNOSTICS`

### Recommended local-fast profile

```env
LOCAL_LLM_MODEL=gemma4:e4b
LOCAL_LLM_VALIDATE_MODEL=true
LOCAL_LLM_TEMPERATURE=0.25
LOCAL_LLM_NUM_PREDICT=256
LOCAL_LLM_RETRY_NUM_PREDICT=512
LOCAL_LLM_STREAM=true
LOCAL_LLM_CONNECT_TIMEOUT_SECONDS=10
LOCAL_LLM_READ_TIMEOUT_SECONDS=150
LOCAL_LLM_NUM_GPU=-1
ENABLE_GEMINI_FALLBACK=false
MAX_CONTEXT_CHARACTERS=2200
RETRIEVAL_TOP_K=3
```

### Troubleshooting

- `HTTP 404 model not found`:
	- Ensure `LOCAL_LLM_MODEL` exactly matches an installed tag from `ollama list`.
	- Example mismatch: configured `qwen3.5:4b` while only `gemma4:e4b` is installed.

- Slow answers / timeout:
	- Lower `LOCAL_LLM_NUM_PREDICT`.
	- Reduce `MAX_CONTEXT_CHARACTERS`.
	- Keep `LOCAL_LLM_STREAM=true`.

- Empty local responses:
	- Check logs for `done_reason`, `eval_count`, `prompt_eval_count`, and local payload preview.
	- Confirm model is valid and loaded with `LOCAL_LLM_VALIDATE_MODEL=true`.

## Notes

- FAISS is currently the primary vector backend and is intended for single-instance use.
- The app is optimized for grounded answers; low-confidence queries should refuse rather than hallucinate.
- Duplicate PDFs are skipped based on content hash.
- Query logs include stage timing (`retrieval_ms`, `prompt_build_ms`, `generation_ms`) for latency tuning.

## License

No license file has been added yet.