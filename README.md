# Enterprise RAG Assistant

Enterprise-focused Retrieval-Augmented Generation (RAG) assistant for PDF knowledge bases, designed for grounded answers, transparent diagnostics, and practical local deployment.

## Overview

This project ingests PDF documents, builds a FAISS-backed vector index, and answers user questions with source-grounded responses.

It includes:
- A FastAPI backend for ingestion, retrieval, generation, verification, and diagnostics.
- A React + TypeScript frontend for upload, chat, streaming responses, and context cards.
- A local-first architecture with optional OpenAI primary routing and local fallback.

## Tech Stack

- Backend: FastAPI, LangChain, FAISS
- Retrieval: Hybrid vector + lexical (BM25-style) fusion with optional reranking
- Embeddings: sentence-transformers via HuggingFace embeddings (CPU/CUDA selectable)
- Generation: OpenAI (optional) and local Ollama fallback
- Frontend: React, TypeScript, Vite
- Storage: Local filesystem for uploads + FAISS index persistence

## Core Capabilities

- Multi-file PDF upload and ingestion
- Duplicate document detection using file hashing
- Structured chunking and metadata enrichment
- Optional Docling-based parsing and vision/table enrichment
- Hybrid retrieval with scoring fusion and reranker gating
- Streaming responses over Server-Sent Events (SSE)
- Source references with snippet previews
- Confidence scoring and retrieval diagnostics
- Post-generation verification (claims/citations consistency checks)
- Knowledge base reset and indexed-files listing endpoints

## High-Level Pipeline

1. Ingestion
- Parse PDF text (structured parser when enabled, legacy fallback otherwise)
- Chunk text with overlap and metadata
- Embed chunks and write vectors into FAISS

2. Retrieval
- Run vector and lexical retrieval branches
- Fuse scores and apply optional reranking
- Select top grounded context chunks

3. Generation
- Build constrained prompt from selected context
- Generate answer via OpenAI or local LLM route
- Stream partial answer tokens/events when using `/query/stream`

4. Verification and Diagnostics
- Validate claims/citations against retrieved context
- Compute confidence signals and include diagnostics in response

## Repository Structure

```text
backend/
  app/
    api/                # FastAPI routes (upload/query)
    services/           # ingestion, retrieval, embedding, vector store, pipeline
    models/             # request/response schemas
    prompts/            # QA prompts
    utils/              # logging utilities
  requirements.txt

frontend/
  src/                  # React UI
  package.json

data/
  uploads/              # uploaded PDF files
  faiss_index/          # persisted FAISS artifacts
```

## Prerequisites

- Windows, macOS, or Linux
- Python 3.10+ (3.11 recommended for longer-term support)
- Node.js 18+
- npm 9+
- Optional: NVIDIA GPU + CUDA-compatible PyTorch for faster embeddings
- Optional: Ollama for local generation route

## Backend Setup

From repository root:

```powershell
python -m venv rag
.\rag\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\backend\requirements.txt
```

Create your environment file:

```powershell
Copy-Item .\backend\.env.example .\backend\.env
```

Update values in `backend/.env` (API keys, model routing flags, embedding device, etc.).

Run backend from `backend/` using the venv interpreter:

```powershell
cd .\backend
python -m uvicorn app.main:app --reload --port 8000
```

Important: prefer `python -m uvicorn` (inside the active venv) to avoid global-Python dependency mismatches.

## Frontend Setup

From repository root:

```powershell
cd .\frontend
npm install
npm run dev
```

Frontend default API target: `http://localhost:8000`.
Override with `VITE_API_URL` if needed.

## API Endpoints

### Health

- `GET /health`
  - Liveness/health check.

### Upload and Knowledge Base

- `POST /upload`
  - Multipart upload for one or more PDF files.
  - Returns per-file status (`success`, `duplicate`, `failed`) plus aggregate counts.

- `GET /knowledge-base/files`
  - Returns persisted indexed file metadata for UI hydration.

- `POST /knowledge-base/reset`
  - Clears uploads, index artifacts, and query cache.

### Query

- `POST /query`
  - Returns full response payload: answer, sources, confidence, diagnostics.

- `POST /query/stream`
  - SSE stream with events:
    - `chunk` (partial text)
    - `done` (final payload)
    - `error` (failure details)

## Configuration Guide

All settings are defined in `backend/app/config.py` and loaded from environment variables.

Common groups:

- Embeddings and device
  - `EMBEDDING_MODEL`
  - `EMBEDDING_DEVICE` (`auto`, `cpu`, `cuda`)
  - `EMBEDDING_BATCH_SIZE`
  - `EMBEDDING_PARALLEL_WORKERS`

- Retrieval and reranking
  - `RETRIEVAL_*`
  - `VECTOR_WEIGHT`, `LEXICAL_WEIGHT`
  - `ENABLE_NEURAL_RERANKER`, `RERANK_*`

- Generation routing
  - `USE_OPENAI`, `OPENAI_*`
  - `LOCAL_LLM_*`
  - `LLM_MAX_TOKENS`

- Verification and diagnostics
  - `ENABLE_VERIFICATION`
  - `VERIFICATION_*`
  - `ENABLE_RETRIEVAL_DIAGNOSTICS`

- Ingestion controls
  - `ENABLE_DOCLING`
  - `ENABLE_VISION_ENRICHMENT`
  - OCR/vision/table toggles and related caps

- Cache and performance
  - `ENABLE_QUERY_CACHE`
  - `QUERY_CACHE_TTL_SECONDS`

## GPU Embeddings (Optional)

To run sentence-transformers on GPU:

1. Install CUDA-enabled PyTorch in the same project venv.
2. Set `EMBEDDING_DEVICE=cuda` in `backend/.env`.
3. Restart backend and verify startup log contains:
   - `Embedding device resolved to: cuda`

If CUDA is unavailable, runtime falls back safely to CPU.

## Logging and Observability

The backend logs:

- Embedding batch progress and total embedding time
- FAISS write time and total `add_documents` time
- Per-file indexing timing:
  - chunking time
  - embed+FAISS time
  - total file ingestion time

This makes ingestion bottlenecks visible without external profiling.

## Troubleshooting

### 1. Backend uses wrong Python environment

Symptom:
- Missing module errors (for example, `No module named pdfplumber`) even after install.

Fix:
- Activate the project venv and run:
  - `python -m uvicorn app.main:app --reload --port 8000`

### 2. FAISS import errors with NumPy 2.x

Symptom:
- `_ARRAY_API not found` or `numpy.core.multiarray failed to import`.

Cause:
- `faiss-cpu==1.8.0` is not compatible with NumPy 2.x ABI.

Fix:
- Use `numpy==1.26.4` (already pinned in requirements).

### 3. CUDA requested but unavailable

Symptom:
- Log warning that CUDA is unavailable and fallback to CPU occurs.

Fix:
- Install matching CUDA-enabled PyTorch wheel
- Verify with:
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

### 4. Reset action appears to hang in UI

Frontend now includes reset-request timeout handling and clear status messages if backend is unreachable.

## Development Workflow

Backend:

```powershell
cd .\backend
python -m uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd .\frontend
npm run dev
```

Open frontend URL from Vite output (usually `http://localhost:5173`).

## Security Notes

- Do not commit real API keys to source control.
- Keep sensitive values in `backend/.env` only.
- Rotate keys immediately if accidentally exposed.

## Current Limitations

- FAISS persistence is local filesystem based (single-node oriented).
- OCR/vision quality depends on document quality and configured model path.
- Python 3.10 remains supported now, but upgrade planning to 3.11+ is recommended.

## License

No license file is currently included in this repository.