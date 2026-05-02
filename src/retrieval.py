import os
import numpy as np
from sentence_transformers import SentenceTransformer
from database import get_connection
from rank_bm25 import BM25Okapi

model = SentenceTransformer("all-MiniLM-L6-v2")

# How many chunks to retrieve per strategy
TOP_K = 5


def chunk_text_helper(text, chunk_size=400, overlap=50):
    """Standalone chunk function used by tests."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def embed_query(query):
    """Convert a question into a 384-dimensional vector."""
    return model.encode(query).tolist()


def retrieve_dense(query):
    """
    Strategy A: Dense vector search.
    Embed the query and find the most semantically similar chunks
    using cosine similarity in pgvector.
    """
    query_embedding = embed_query(query)

    conn = get_connection()
    cursor = conn.cursor()

    # The <=> operator is pgvector's cosine distance
    # We order by distance ascending — smaller distance = more similar
    cursor.execute("""
        SELECT 
            id,
            topic,
            url,
            chunk_index,
            content,
            1 - (embedding <=> %s::vector) as similarity
        FROM knowledge_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, TOP_K))

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return [dict(row) for row in results]


def retrieve_bm25(query, candidates=None):
    """
    BM25 keyword search over the knowledge base.
    BM25 scores documents based on term frequency and inverse document frequency.
    Returns top K results ranked by keyword relevance.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, topic, url, chunk_index, content FROM knowledge_chunks")
    all_chunks = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    # Tokenize — split each chunk into individual words (lowercase)
    tokenized_corpus = [chunk["content"].lower().split() for chunk in all_chunks]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its BM25 score and sort
    scored = sorted(
        zip(all_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_chunks = []
    for chunk, score in scored[:TOP_K]:
        chunk["bm25_score"] = score
        top_chunks.append(chunk)

    return top_chunks


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """
    Strategy B: Combine dense and BM25 results using Reciprocal Rank Fusion.

    RRF formula: score = sum of 1/(k + rank) across all ranking lists.

    Why k=60? It's a constant that dampens the impact of very high rankings
    — prevents one method from completely dominating just because it ranked
    something #1. Empirically validated in the original RRF paper.

    A chunk that ranks #2 in dense AND #3 in BM25 scores higher than
    a chunk that ranks #1 in only one list. This is the key insight —
    agreement between two independent methods is a strong signal.
    """
    scores = {}
    chunk_data = {}

    # Score from dense ranking
    for rank, chunk in enumerate(dense_results):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_data[chunk_id] = chunk

    # Score from BM25 ranking
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_data[chunk_id] = chunk

    # Sort by combined RRF score descending
    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    results = []
    for chunk_id in ranked_ids[:TOP_K]:
        chunk = chunk_data[chunk_id]
        chunk["rrf_score"] = scores[chunk_id]
        results.append(chunk)

    return results


def retrieve_hybrid(query):
    """Strategy B: Hybrid retrieval using RRF to combine dense + BM25."""
    dense_results = retrieve_dense(query)
    bm25_results = retrieve_bm25(query)
    return reciprocal_rank_fusion(dense_results, bm25_results)


def compute_confidence(results, strategy):
    """
    Confidence based on cosine similarity scores.
    Dense results carry similarity directly.
    Hybrid results don't — so we check if any dense similarity
    scores were attached, otherwise fall back to RRF-based estimate.
    """
    if not results:
        return 0.0

    # Try to get actual cosine similarity scores first
    similarity_scores = [r.get("similarity", None) for r in results]
    similarity_scores = [s for s in similarity_scores if s is not None]

    if similarity_scores:
        return round(float(np.mean(similarity_scores)), 4)

    # Hybrid results have rrf_score not similarity
    # RRF max theoretical score for rank 1 from both lists = 2/(60+1) = 0.0328
    # We normalize against that maximum
    rrf_scores = [r.get("rrf_score", 0) for r in results]
    max_theoretical = 2 / (60 + 1)
    normalized = [s / max_theoretical for s in rrf_scores]
    return round(float(np.mean(normalized)), 4)