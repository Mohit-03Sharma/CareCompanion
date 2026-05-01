import sys
sys.path.insert(0, 'src')

from retrieval import retrieve_hybrid, retrieve_dense, compute_confidence
from llm import generate_answer
from safety import check_safety, check_confidence
from database import get_connection


def run_pipeline(query, strategy="hybrid"):
    """
    Full pipeline: safety check → retrieve → confidence check → generate.
    
    Returns a dict with answer, sources, confidence, and safety metadata.
    """

    # Step 1 — Safety layer: check before doing anything else
    is_safe, safety_category, safety_response = check_safety(query)
    if not is_safe:
        log_query(query, None, strategy, 0.0, True, safety_category, safety_response)
        return {
            "answer": safety_response,
            "sources": [],
            "confidence": 0.0,
            "safety_flagged": True,
            "safety_category": safety_category,
            "strategy": strategy
        }

    # Step 2 — Retrieve relevant chunks
    if strategy == "hybrid":
        chunks = retrieve_hybrid(query)
        # Get dense results separately for accurate confidence scoring
        dense_chunks = retrieve_dense(query)
        confidence = compute_confidence(dense_chunks, "dense")
    else:
        chunks = retrieve_dense(query)
        confidence = compute_confidence(chunks, "dense")
        
    # Step 3 — Confidence gate: don't answer if retrieval quality is low
    confident_enough, low_conf_response = check_confidence(confidence)
    if not confident_enough:
        log_query(query, None, strategy, confidence, False, "low_confidence", low_conf_response)
        return {
            "answer": low_conf_response,
            "sources": [],
            "confidence": confidence,
            "safety_flagged": False,
            "safety_category": "low_confidence",
            "strategy": strategy
        }

    # Step 4 — Generate grounded answer
    answer, sources = generate_answer(query, chunks)

    # Step 5 — Log everything for drift monitoring later
    log_query(query, None, strategy, confidence, False, None, answer)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "safety_flagged": False,
        "safety_category": None,
        "strategy": strategy
    }


def log_query(query, query_embedding, strategy, confidence, safety_flagged, safety_category, response):
    """Write every query to the database for drift monitoring in Week 5."""
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