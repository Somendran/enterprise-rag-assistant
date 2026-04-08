# Enterprise RAG Assistant

FastAPI + React application for querying internal PDFs with a retrieval-augmented generation pipeline.

## What it does

- Upload PDFs into a FAISS-backed knowledge base.
- Ask natural-language questions against indexed documents.
- Return grounded answers with source citations.
- Expose confidence scores and retrieval diagnostics for quality tuning.
- Prevent duplicate uploads by hashing document content.

## Current architecture

- Backend: FastAPI, LangChain, FAISS, Gemini 2.5 Flash.
- Frontend: React 19 + TypeScript + Vite.
- Embeddings: Gemini embeddings or local Hugging Face fallback.
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
- A valid Google API key for Gemini access

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a backend `.env` file from `backend/.env.example` and set at least:

- `GOOGLE_API_KEY`
- `EMBEDDING_MODEL`
- `LLM_MODEL`

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

- `RETRIEVAL_TOP_K`
- `RETRIEVAL_CANDIDATE_K`
- `RETRIEVAL_LOW_CONFIDENCE_THRESHOLD`
- `ANSWER_LOW_CONFIDENCE_THRESHOLD`
- `ENABLE_RETRIEVAL_FALLBACK`
- `ENABLE_RETRIEVAL_DIAGNOSTICS`

## Notes

- FAISS is currently the primary vector backend and is intended for single-instance use.
- The app is optimized for grounded answers; low-confidence queries should refuse rather than hallucinate.
- Duplicate PDFs are skipped based on content hash.

## License

No license file has been added yet.