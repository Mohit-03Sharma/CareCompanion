from retrieval import retrieve_hybrid, retrieve_dense, compute_confidence
from llm import generate_answer
from safety import check_safety, check_confidence
from cache import get_cached_response, cache_response
from database import get_connection


def run_pipeline(query, strategy="hybrid"):
    """Safety check → cache lookup → retrieve → confidence gate → generate."""

    # Step 1 — Safety check first, before anything else
    is_safe, safety_category, safety_response = check_safety(query)
    if not is_safe:
        log_query(query, strategy, 0.0, True, safety_category, safety_response)
        return {
            "answer": safety_response,
            "sources": [],
            "confidence": 0.0,
            "safety_flagged": True,
            "safety_category": safety_category,
            "strategy": strategy,
            "cache_hit": False
        }

    # Step 2 — Check cache before hitting retrieval or LLM
    cached = get_cached_response(query)
    if cached:
        return cached

    # Step 3 — Retrieve relevant chunks
    if strategy == "hybrid":
        chunks = retrieve_hybrid(query)
        dense_chunks = retrieve_dense(query)
        confidence = compute_confidence(dense_chunks, "dense")
    else:
        chunks = retrieve_dense(query)
        confidence = compute_confidence(chunks, "dense")

    # Step 4 — Confidence gate
    confident_enough, low_conf_response = check_confidence(confidence)
    if not confident_enough:
        log_query(query, strategy, confidence, False, "low_confidence", low_conf_response)
        return {
            "answer": low_conf_response,
            "sources": [],
            "confidence": confidence,
            "safety_flagged": False,
            "safety_category": "low_confidence",
            "strategy": strategy,
            "cache_hit": False
        }

    # Step 5 — Generate answer
    answer, sources = generate_answer(query, chunks)

    # Step 6 — Cache the result for future similar queries
    cache_response(query, answer, sources, confidence, strategy)

    # Step 7 — Log the query
    log_query(query, strategy, confidence, False, None, answer)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "safety_flagged": False,
        "safety_category": None,
        "strategy": strategy,
        "cache_hit": False
    }


def log_query(query, strategy, confidence, safety_flagged, safety_category, response):
    """Write every query to the database for drift monitoring."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_logs
                (query, retrieval_strategy, confidence_score,
                 safety_flagged, safety_category, response)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (query, strategy, confidence, safety_flagged, safety_category, response))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Logging error: {e}")