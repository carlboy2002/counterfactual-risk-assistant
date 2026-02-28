"""
rag_engine.py
-------------
Handles the full RAG pipeline:
  1. Semantic chunking of the cleaned 10-K text
  2. Embedding via OpenAI text-embedding-3-small
  3. Storage and retrieval from a local ChromaDB instance
  4. Multi-query retrieval router for financial, operational, and regulatory risks

Design principle: The RAG layer ONLY retrieves candidate evidence chunks.
It does NOT interpret, summarize, or classify risks — that is the LLM's job.
"""

import hashlib
import logging
import re
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Risk-focused retrieval queries ---
# These queries are used to semantically search the vector DB for
# candidate evidence chunks that MIGHT contain risk information.
# We use three semantic angles to maximize recall across the full 10-K.
RISK_RETRIEVAL_QUERIES = {
    "financial": [
        "revenue decline liquidity credit default debt covenant risk",
        "interest rate foreign exchange currency volatility exposure",
        "impairment goodwill write-down asset valuation uncertainty",
        "cash flow negative working capital going concern",
    ],
    "operational": [
        "supply chain disruption manufacturing defect product recall",
        "key personnel dependency talent retention workforce",
        "cybersecurity data breach information security incident",
        "business continuity disaster recovery operational failure",
        "competition market share pricing pressure competitive advantage",
    ],
    "regulatory": [
        "regulatory compliance government investigation legal proceeding",
        "environmental liability climate change regulation ESG",
        "antitrust intellectual property patent litigation",
        "sanctions export control international operations geopolitical",
        "tax uncertainty legislation policy change compliance risk",
    ],
}

# Chunking parameters — tuned for financial documents
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # overlap to preserve context across chunk boundaries
TOP_K_PER_QUERY = 3      # number of chunks retrieved per individual query


def build_vector_store(
    clean_text: str,
    ticker: str,
    year: int,
    openai_api_key: str,
    collection_name: str = None,
) -> Chroma:
    """
    Chunk the full 10-K text, embed it, and store it in ChromaDB (in-memory).
    Returns the Chroma vector store object.

    Args:
        clean_text: Full cleaned 10-K text
        ticker: Stock ticker (used for chunk metadata and collection naming)
        year: Fiscal year
        openai_api_key: OpenAI API key for embeddings
        collection_name: Optional override for the Chroma collection name
    """
    if not collection_name:
        collection_name = f"10k_{ticker.lower()}_{year}"

    logger.info(f"Building vector store for collection: {collection_name}")

    # --- Step 1: Semantic Chunking ---
    # RecursiveCharacterTextSplitter respects paragraph/sentence boundaries
    # better than a fixed-size splitter, producing more coherent chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
    )
    raw_chunks = splitter.split_text(clean_text)
    logger.info(f"Split into {len(raw_chunks)} chunks")

    # --- Step 2: Create LangChain Documents with rich metadata ---
    documents = []
    for i, chunk_text in enumerate(raw_chunks):
        # Deterministic chunk_id: hash of ticker+year+index
        chunk_id = f"{ticker.upper()}_{year}_chunk_{i:04d}"
        # Also store a content hash for deduplication detection
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]

        doc = Document(
            page_content=chunk_text,
            metadata={
                "chunk_id": chunk_id,
                "chunk_index": i,
                "ticker": ticker.upper(),
                "year": year,
                "content_hash": content_hash,
                "char_count": len(chunk_text),
                "source": f"SEC 10-K {ticker.upper()} FY{year}",
            },
        )
        documents.append(doc)

    # --- Step 3: Embed and Store in ChromaDB ---
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    # Use ephemeral in-memory Chroma (no persistence needed per session)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
    )

    logger.info(f"Vector store built with {len(documents)} documents")
    return vector_store


def retrieve_risk_candidates(
    vector_store: Chroma,
    top_k: int = TOP_K_PER_QUERY,
) -> list[dict[str, Any]]:
    """
    The RAG Retrieval Router.
    Runs multiple semantically-diverse queries across risk categories
    (financial, operational, regulatory) to maximize recall.

    This function intentionally does NOT filter or classify the retrieved
    chunks — it only returns candidate evidence for the LLM to evaluate.

    Returns a deduplicated list of chunk dicts:
        [{"chunk_id": ..., "text": ..., "category": ..., "score": ...}]
    """
    seen_chunk_ids = set()
    all_candidates = []

    for category, queries in RISK_RETRIEVAL_QUERIES.items():
        for query in queries:
            try:
                # similarity_search_with_score returns (Document, float) tuples
                results = vector_store.similarity_search_with_score(query, k=top_k)
                for doc, score in results:
                    chunk_id = doc.metadata.get("chunk_id", str(uuid.uuid4()))
                    if chunk_id in seen_chunk_ids:
                        continue  # Deduplicate across queries
                    seen_chunk_ids.add(chunk_id)
                    all_candidates.append(
                        {
                            "chunk_id": chunk_id,
                            "text": doc.page_content,
                            "category": category,
                            "retrieval_score": round(float(score), 4),
                            "metadata": doc.metadata,
                        }
                    )
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query[:40]}...': {e}")

    # Sort by retrieval score (lower = more similar for cosine distance)
    all_candidates.sort(key=lambda x: x["retrieval_score"])
    logger.info(f"Retrieved {len(all_candidates)} unique candidate chunks")
    return all_candidates


def format_chunks_for_prompt(candidates: list[dict[str, Any]], max_chunks: int = 20) -> str:
    """
    Format retrieved chunks into a structured string for the LLM prompt.
    Each chunk is labeled with its chunk_id so the LLM can cite sources.
    Limits to max_chunks to stay within token budget.
    """
    selected = candidates[:max_chunks]
    parts = []
    for i, c in enumerate(selected):
        parts.append(
            f"[CHUNK_ID: {c['chunk_id']} | Category: {c['category'].upper()} | Score: {c['retrieval_score']}]\n"
            f"{c['text']}\n"
            f"{'—' * 60}"
        )
    return "\n\n".join(parts)
