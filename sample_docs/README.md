# Sample Documents

Generate local demo PDFs:

```powershell
cd D:\enterprise-rag-assistant
python .\sample_docs\generate_sample_pdfs.py
```

Then upload the generated PDFs through the RAGiT UI at `http://localhost:5173`.

The eval fixture in `evals/questions.json` is aligned to these sample documents. After upload, run:

```powershell
python .\evals\run_eval.py --live --api-url http://localhost:8000
```

If `APP_API_KEY` is configured, pass `--api-key`.
