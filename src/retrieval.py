import os
import numpy as np
from sentence_transformers import SentenceTransformer
from database import get_connection
from rank_bm25 import BM25Okapi

model = SentenceTransformer("all-MiniLM-L6-v2")

# How many chunks to retrieve per strategy
TOP_K = 5


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


def retrieve_dense_by_ids(chunk_ids):
    """Fetch similarity scores for specific chunk IDs against a stored query."""
    # Since we can't re-run the vector query here, we return the
    # dense results that were already computed during hybrid retrieval
    # This is why pipeline.py will pass both results through
    return []

def compute_confidence(results, strategy):
    """Confidence score based on absolute retrieval similarity, not relative ranking."""
    if not results:
        return 0.0

    if strategy == "dense":
        scores = [r.get("similarity", 0) for r in results]
        return round(float(np.mean(scores)), 4)
    else:
        # For hybrid, use the dense similarity scores of the returned chunks
        # RRF scores are relative rankings, not absolute similarity measures
        # We need absolute similarity to make a meaningful confidence judgment
        dense_results = retrieve_dense_by_ids(
            [r["id"] for r in results]
        )
        if not dense_results:
            return 0.0
        scores = [r.get("similarity", 0) for r in dense_results]
        return round(float(np.mean(scores)), 4)