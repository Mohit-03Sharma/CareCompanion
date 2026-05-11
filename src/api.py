from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_pipeline

app = FastAPI(title="CareCompanion API")


class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    safety_flagged: bool
    safety_category: str | None
    strategy: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    result = run_pipeline(request.query, request.strategy)
    return result


@app.get("/experiment-results")
def experiment_results():
    from database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT strategy,
               COUNT(*) as questions,
               ROUND(AVG(relevance_score)::numeric, 4) as mean_score
        FROM experiment_runs
        WHERE experiment_name = 'dense_vs_hybrid'
        GROUP BY strategy
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return {"results": rows}


@app.get("/drift-status")
def drift_status():
    from database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as total_queries,
               ROUND(AVG(confidence_score)::numeric, 4) as mean_confidence,
               SUM(CASE WHEN safety_flagged THEN 1 ELSE 0 END) as safety_flags
        FROM query_logs
    """)
    row = dict(cursor.fetchone())
    cursor.close()
    conn.close()
    return row