"""
qa_prompt.py
────────────
Defines the strict, grounded prompt used for the RAG question-answering chain.

Design principles:
- The LLM is instructed to ONLY use the provided context.
- If no relevant information is found, it must say so honestly.
- Sources are always referenced in the answer.
- This prevents hallucination and keeps the assistant trustworthy.
"""

from langchain.prompts import PromptTemplate


# The {context} placeholder will be filled with retrieved document chunks.
# The {question} placeholder will be filled with the user's question.
QA_PROMPT_TEMPLATE = """You are an intelligent internal knowledge assistant for an enterprise.
Your job is to answer employee questions strictly based on the provided company documents.

RULES:
1. Only use information from the CONTEXT below. Do not use any outside knowledge.
2. If the answer cannot be found in the context, respond exactly with:
   "I don't know based on the available documents."
3. Always cite which document and page you used at the end of your answer.
4. Be concise and professional. Avoid unnecessary filler.
5. If multiple documents are relevant, synthesize them and cite all sources.

─────────────────────────────────────────
CONTEXT (retrieved document excerpts):
{context}
─────────────────────────────────────────

QUESTION: {question}

ANSWER (cite sources at the end):"""


QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)
