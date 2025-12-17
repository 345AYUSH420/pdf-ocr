import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from vision_ocr_loader import load_pdf_with_vision_ocr, read_ocr_output_documents
from langchain_core.documents import Document
from typing import Optional

# from ocr_loader import load_pdf_with_ocr
from vector_store import setup_vector_store

app = FastAPI(title="PDF RAG API")
rag_chain = None

def _as_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _format_context(docs) -> str:
    # OpenRouter free tiers can have very small prompt token limits.
    # Keep context tight to avoid 402 "Prompt tokens limit exceeded".
    max_chars = int(os.getenv("MAX_CONTEXT_CHARS", "1000"))
    if not docs:
        return ""
    text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    return text[:max_chars]


def _build_rag_chain():
    """Build the RAG chain once at startup (reuses Pinecone embeddings)."""
    load_dotenv()

    input_gcs_uri = os.getenv("INPUT_GCS_URI", "gs://ocr-extraction-bucket/MADHAVA.pdf.pdf")
    output_gcs_uri = os.getenv("OUTPUT_GCS_URI", "gs://ocr-extracted-bucket/output/")
    index_name = os.getenv("PINECONE_INDEX", "books")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")

    # If embeddings already exist in Pinecone, skip OCR + embedding work.
    # Set FORCE_REINDEX=1 to force OCR + (re)upload.
    force_reindex = _as_bool(os.getenv("FORCE_REINDEX"))
    # Default OFF: if you already have embeddings, don't OCR again by mistake.
    allow_ocr_fallback = _as_bool(os.getenv("ALLOW_OCR_FALLBACK", "0"))

    retriever = None
    if not force_reindex:
        try:
            retriever = setup_vector_store(docs=None, index_name=index_name, namespace=namespace)
            print("✅ Using existing Pinecone embeddings — skipping OCR.")
        except ValueError:
            retriever = None

    if retriever is None:
        if not allow_ocr_fallback and not force_reindex:
            raise RuntimeError(
                f"Pinecone index '{index_name}' not found (namespace '{namespace}'). "
                "Set ALLOW_OCR_FALLBACK=1 to build embeddings, or create the index first."
            )

        # Ensure OCR is available (runs OCR once, then uses cache on subsequent runs)
        ocr_text = load_pdf_with_vision_ocr(
            input_gcs_uri=input_gcs_uri,
            output_gcs_uri=output_gcs_uri,
        )

        # Prefer page-level Documents (better retrieval + safer embedding). If for some
        # reason the output JSON isn't present anymore, fall back to cached full text.
        docs = read_ocr_output_documents(output_gcs_uri, source_uri=input_gcs_uri)
        if not docs:
            docs = [Document(page_content=ocr_text, metadata={"source": input_gcs_uri})]

        retriever = setup_vector_store(
            docs,
            force_reindex=force_reindex,
            index_name=index_name,
            namespace=namespace,
        )

    # LLM on OpenRouter (revert back to OpenRouter)
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo"),
        max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a multilingual RAG assistant that MUST translate answers to match the user's question language. "
                "Step 1: Use the provided context to find the answer. "
                "Step 2: Translate the answer to the user's language. "
                "Rules: "
                "- If the question is in English, translate your answer to English. "
                "- If the question is in Hindi, answer in Hindi. "
                "- Always provide the answer in the user's language, NOT the source language. "
                "If the answer is not in context, reply: 'Not found in the provided text.'",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Context:\n{context}\n\n"
                "Answer strictly using the context.",
            ),
        ]
    )

    parser = StrOutputParser()

    return (
        {
            "context": retriever | RunnableLambda(_format_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag_chain is not None}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized yet.")
    try:
        answer = await run_in_threadpool(rag_chain.invoke, payload.question)
        return AskResponse(answer=answer)
    except Exception as e:
        msg = str(e)
        # OpenRouter can return errors for missing credits / limits.
        if "Error code: 402" in msg or "credits" in msg:
            raise HTTPException(
                status_code=402,
                detail=(
                    "OpenRouter credits/limit error. "
                    "Add credits in OpenRouter or reduce context size. "
                    "Raw error: " + msg
                ),
            )
        raise HTTPException(status_code=500, detail=msg)


@app.on_event("startup")
def _startup():
    global rag_chain
    rag_chain = _build_rag_chain()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=_as_bool(os.getenv("RELOAD", "0")),
    )
