from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env
load_dotenv()

def _as_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_total_vector_count(index) -> int:
    """Best-effort vector count across all namespaces."""
    stats = index.describe_index_stats()
    # Pinecone client may return a dict or an object, depending on version.
    if hasattr(stats, "total_vector_count") and stats.total_vector_count is not None:
        return int(stats.total_vector_count)
    if isinstance(stats, dict):
        for key in ("total_vector_count", "totalVectorCount", "total_vector_count".upper()):
            if key in stats and stats[key] is not None:
                return int(stats[key])
    # If we can't determine, assume empty so we can (re)build when docs provided.
    return 0


def _get_namespace_vector_count(index, namespace: str) -> int:
    """Best-effort vector count for a specific namespace."""
    stats = index.describe_index_stats()

    namespaces = None
    if hasattr(stats, "namespaces"):
        namespaces = getattr(stats, "namespaces")
    elif isinstance(stats, dict):
        namespaces = stats.get("namespaces")

    if isinstance(namespaces, dict):
        ns_stats = namespaces.get(namespace) or {}
        if isinstance(ns_stats, dict):
            for key in ("vector_count", "vectorCount"):
                if key in ns_stats and ns_stats[key] is not None:
                    return int(ns_stats[key])
        if hasattr(ns_stats, "vector_count") and getattr(ns_stats, "vector_count") is not None:
            return int(getattr(ns_stats, "vector_count"))

    # fallback to total count
    return _get_total_vector_count(index)


def setup_vector_store(
    docs=None,
    *,
    index_name: str = "books",
    namespace: str = "default",
    force_reindex: Optional[bool] = None,
):
    """Return a retriever backed by Pinecone.

    - If `docs` is None: loads an existing index (no OCR/embedding generation).
    - If `docs` is provided: creates the index if needed and uploads chunks.
    - If `force_reindex` is True: clears the namespace then re-uploads chunks.
    """
    if force_reindex is None:
        force_reindex = _as_bool(os.getenv("FORCE_REINDEX"))

    # Embeddings on OpenRouter (revert back to OpenRouter)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        # Important: OpenRouter (and some OpenAI-compatible gateways) can error if
        # you send too many texts in one embeddings request.
        chunk_size=16,
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_names = []
    for i in pc.list_indexes():
        # Pinecone may return dicts or objects depending on version.
        if isinstance(i, dict) and "name" in i:
            index_names.append(i["name"])
        elif hasattr(i, "name"):
            index_names.append(i.name)

    index_exists = index_name in index_names

    if not index_exists:
        if docs is None:
            raise ValueError(
                f"Pinecone index '{index_name}' does not exist yet. Provide docs to build it."
            )

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-large",
            chunk_size=3000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace,
        )
    else:
        vectorstore = PineconeVectorStore.from_existing_index(
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace,
        )

        # Only upload documents if provided (avoids OCR + embedding work on startup).
        if docs is not None:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="text-embedding-3-large",
                chunk_size=4000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)

            index = pc.Index(index_name)
            if force_reindex:
                print(f"🧹 FORCE_REINDEX=1 — clearing namespace '{namespace}'...")
                index.delete(delete_all=True, namespace=namespace)

            existing_count = _get_namespace_vector_count(index, namespace)
            new_count = len(chunks)

            if existing_count == 0:
                print(f"📝 Adding {new_count} chunks to Pinecone index '{index_name}'...")
                vectorstore.add_documents(documents=chunks)
            else:
                print(
                    f"⚠️ Index already has {existing_count} vectors. Skipping upload to avoid duplicates."
                )

    # Keep k small to avoid blowing up the LLM prompt (OpenRouter free tiers can
    # have low prompt-token limits).
    print(vectorstore.as_retriever(search_kwargs={"k": 5}))
    return vectorstore.as_retriever(search_kwargs={"k": 5})
