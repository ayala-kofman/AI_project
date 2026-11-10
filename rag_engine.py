"""
RAG Engine - Core logic for the organizational knowledge bot.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import TokenTextSplitter

# 🟢 CHANGED: שימוש באינטגרציה הייעודית של Gemini
from llama_index.llms.google_genai import GoogleGenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "org_knowledge"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3


def _configure_llama_index():
    """
    - LLM: Gemini 3.8 Flash
    - Embeddings: local multilingual model
    """

    # 🟢 CHANGED: חיבור ישיר ל-Gemini במקום OpenAI
    Settings.llm = GoogleGenAI(
        model="gemini-3.8-flash",
        api_key=GEMINI_API_KEY,
    )

    # 🟢 CHANGED: השארת מודל Embedding אחד בלבד
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small"
    )


def _get_splitter():
    return TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def build_index() -> VectorStoreIndex:
    """Load documents from data/ and build the ChromaDB index."""
    _configure_llama_index()

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise FileNotFoundError(
            "No documents found in the data/ folder. "
            "Please add your organizational documents first."
        )

    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[_get_splitter()],
        show_progress=True,
    )
    return index


def load_index() -> VectorStoreIndex:
    """Load an already-built index from ChromaDB (no re-embedding)."""
    _configure_llama_index()
    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def _expand_query(raw_query: str) -> str:
    """Expand short queries before embedding to improve retrieval."""
    if len(raw_query.split()) >= 8:
        return raw_query

    prompt = (
        "You are a search query optimizer. "
        "Expand the following short question into a detailed search query "
        "that will help retrieve relevant documents. "
        "Return ONLY the expanded query, nothing else.\n\n"
        f"Original question: {raw_query}\n"
        "Expanded query:"
    )
    try:
        expanded = Settings.llm.complete(prompt).text.strip()
        return expanded if expanded else raw_query
    except Exception:
        return raw_query


def answer_question(index: VectorStoreIndex, user_question: str) -> dict:
    """Return the answer and source info for a user question."""
    expanded = _expand_query(user_question)

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact",
    )

    response = query_engine.query(expanded)

    sources = []
    for node in response.source_nodes:
        fname = node.metadata.get("file_name", "unknown")
        score = round(node.score, 3) if node.score else "N/A"
        sources.append(f"{fname} (score: {score})")

    return {
        "answer": str(response),
        "sources": sources,
        "expanded_query": expanded if expanded != user_question else None,
    }


def index_is_empty() -> bool:
    """Return True if no vectors have been stored yet."""
    try:
        col = _get_chroma_collection()
        return col.count() == 0
    except Exception:
        return True