import json
import numpy as np
from sentence_transformers import SentenceTransformer
from database import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.95


def get_cached_response(query):
    """
    Check if a semantically similar query has been answered before.
    Returns cached result if similarity > threshold, else None.
    """
    query_embedding = model.encode(query).tolist()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            query_text,
            answer,
            sources,
            confidence,
            strategy,
            1 - (query_embedding <=> %s::vector) as similarity
        FROM response_cache
        ORDER BY query_embedding <=> %s::vector
        LIMIT 1
    """, (query_embedding, query_embedding))

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return None

    row = dict(row)
    if row["similarity"] < SIMILARITY_THRESHOLD:
        return None

    return {
        "answer": row["answer"],
        "sources": row["sources"] if row["sources"] else [],
        "confidence": row["confidence"],
        "strategy": row["strategy"],
        "cache_hit": True,
        "cache_similarity": round(row["similarity"], 4),
        "safety_flagged": False,
        "safety_category": None
    }


def cache_response(query, answer, sources, confidence, strategy):
    """Store a query-response pair in the cache."""
    query_embedding = model.encode(query).tolist()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO response_cache
            (query_embedding, query_text, answer, sources, confidence, strategy)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        query_embedding,
        query,
        answer,
        json.dumps(sources),
        confidence,
        strategy
    ))

    conn.commit()
    cursor.close()
    conn.close()