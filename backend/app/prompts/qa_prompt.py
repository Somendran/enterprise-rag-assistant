"""
qa_prompt.py
────────────
Defines the strict, grounded prompt used for the RAG question-answering chain.

Design principles:
- The LLM is instructed to ONLY use the provided context.
- If no relevant information is found, it must say so honestly.
- Source references are returned separately by the API response schema.
- This prevents hallucination and keeps the assistant trustworthy.
"""

from langchain.prompts import PromptTemplate


# The {context} placeholder will be filled with retrieved document chunks.
# The {question} placeholder will be filled with the user's question.
QA_PROMPT_TEMPLATE = """You are a helpful enterprise assistant.

Task:
- Summarize the provided CONTENT to answer the QUESTION.
- Use only information present in CONTENT.
- If information is partial, provide a best-effort summary and mark uncertainty briefly.
- Do not include reasoning traces.

Output format (strict):
## Executive Summary
- 2 to 3 concise bullets

## Key Facts
- 3 to 5 concise bullets with concrete facts or numbers when available

## Risks / Limitations
- 1 to 2 concise bullets about missing or uncertain information

If no relevant facts exist in CONTENT, output exactly: I don't know.

CONTENT:
{context}

QUESTION:
{question}

ANSWER:"""


QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)
