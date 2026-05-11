# CareCompanion

![CI Pipeline](https://github.com/Mohit-03Sharma/CareCompanion/actions/workflows/ci.yml/badge.svg)

A dual-interface clinical intelligence platform — a patient-facing health Q&A grounded in NIH MedlinePlus content, and a clinician-facing note extraction tool that parses unstructured clinical notes into structured JSON validated against medical knowledge.

Built to demonstrate the architecture health-tech companies use when they need controlled, auditable, API-accessible health information delivery — as opposed to a general-purpose chatbot.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│         Patient Q&A Page  │  Clinician Extractor Page   │
└──────────────┬────────────────────────┬─────────────────┘
               │                        │
               ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Backend                      │
│   /ask  /extract-note  /experiment-results  /drift-status│
└──────────────┬────────────────────────┬─────────────────┘
               │                        │
       ┌───────▼───────┐       ┌────────▼────────┐
       │  RAG Pipeline  │       │Note Extractor   │
       │                │       │                 │
       │ Safety Check   │       │ pdfplumber/text │
       │ Cache Lookup   │       │ Groq LLM        │
       │ Dense/Hybrid   │       │ NIH Validation  │
       │ Retrieval      │       │ JSON Output     │
       │ Groq LLM       │       └─────────────────┘
       └───────┬────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│              PostgreSQL + pgvector                       │
│                                                          │
│  knowledge_chunks  │  query_logs  │  response_cache      │
│  experiment_runs   │                                     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Knowledge base | 1,318 chunks across 46 NIH MedlinePlus topics |
| Dense retrieval mean relevance | 0.7598 |
| Hybrid retrieval mean relevance | 0.7788 |
| A/B p-value | 0.298 — no statistically significant difference |
| Semantic cache speedup | ~22x faster on cache hits (1.57s → 0.07s) |
| Safety layer | 3 categories — emergency, dosage, diagnosis |
| Unit tests | 8 passing, enforced on every push via GitHub Actions |
| API endpoints | 6 — /health, /ask, /extract-note, /experiment-results, /drift-status, /knowledge-topics |

---

## Features

**Patient Interface**
- Conversational health Q&A grounded exclusively in NIH MedlinePlus content
- Hybrid retrieval combining dense vector search (pgvector) and BM25 keyword search via Reciprocal Rank Fusion
- Three-layer safety system: emergency pattern interception, out-of-scope blocking (dosage, diagnosis), confidence gating
- Every query logged with confidence score, strategy, and safety metadata for drift monitoring
- Semantic caching — semantically similar queries served from cache at ~100ms

**Clinician Interface**
- Accepts pasted text or PDF upload of clinical notes
- Extracts structured fields: chief complaint, diagnoses, medications (with new medication flagging), follow-up actions, warning signs, dietary instructions
- Validates extracted diagnoses against NIH knowledge base — flags conditions not found
- Downloads structured output as JSON

**Experimentation**
- A/B framework comparing dense vs hybrid retrieval on 20 health questions with ground truth answers
- Automated relevance scoring via cosine similarity between generated answers and ground truth
- Paired t-test with p-value and 95% confidence interval
- All runs logged to PostgreSQL and tracked via MLflow

**Monitoring**
- Evidently AI drift reports comparing query embedding distributions across time windows
- Query log analytics exposed via /drift-status endpoint

**Infrastructure**
- GitHub Actions CI/CD: pytest + flake8 on every push to main
- Docker + docker-compose: full stack spins up with one command
- FastAPI backend with auto-generated docs at /docs

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector database | pgvector on PostgreSQL 16 |
| Keyword search | BM25 (rank-bm25) |
| LLM | Llama 3.3 70B via Groq API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Monitoring | Evidently AI |
| Experiment tracking | MLflow |
| PDF parsing | pdfplumber |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |

---

## Setup

**Prerequisites:** Python 3.12, Docker, Git

**1. Clone and install**
```bash
git clone https://github.com/Mohit-03Sharma/CareCompanion.git
cd CareCompanion
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at console.groq.com
```

**3. Start the database**
```bash
docker compose up -d
python src/database.py
```

**4. Ingest knowledge base**
```bash
python src/ingest_medlineplus.py
```

**5. Run the application**
```bash
# Terminal 1 — API
cd src && uvicorn api:app --reload --port 8000

# Terminal 2 — UI
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
carecompanion/
├── src/
│   ├── database.py          # Schema setup — all PostgreSQL tables
│   ├── ingest_medlineplus.py # NIH MedlinePlus ingestion pipeline
│   ├── retrieval.py         # Dense + hybrid BM25/RRF retrieval
│   ├── llm.py               # Groq LLM integration
│   ├── safety.py            # Three-layer safety system
│   ├── cache.py             # Semantic caching layer
│   ├── pipeline.py          # End-to-end query pipeline
│   ├── extractor.py         # Clinical note extraction
│   ├── experiment.py        # A/B experimentation framework
│   ├── monitoring.py        # Evidently AI drift detection
│   └── api.py               # FastAPI application
├── pages/
│   ├── patient.py           # Patient Q&A Streamlit page
│   └── clinician.py         # Clinician extractor Streamlit page
├── tests/
│   └── test_pipeline.py     # 8 unit tests — safety + retrieval
├── notebooks/
│   └── exploration.ipynb    # A/B results, confidence distribution, KB coverage
├── data/
│   └── sample_notes/        # Sample clinical notes for testing
├── .github/workflows/
│   └── ci.yml               # GitHub Actions — test + lint on every push
├── app.py                   # Streamlit home page
├── docker-compose.yml       # PostgreSQL + pgvector container
└── requirements.txt
```

---

## Authentication Note

The current deployment uses a single-interface application with two pages. In a production deployment, role-based authentication would separate patient and clinician access:

- Patients authenticate via email/password — access patient Q&A only
- Clinicians authenticate via institutional SSO — access both interfaces
- JWT tokens with role claims stored in the existing PostgreSQL session infrastructure
- All queries attributed to authenticated users for full audit trail

---

## A/B Experiment Details

Two retrieval strategies were evaluated on 20 health questions with ground truth answers sourced from the NIH knowledge base:

- **Strategy A — Dense:** embed query, find top-5 chunks by cosine similarity in pgvector
- **Strategy B — Hybrid:** combine dense similarity and BM25 keyword scores via Reciprocal Rank Fusion (k=60)

Answer quality was scored automatically by computing cosine similarity between the generated answer and the ground truth answer using the same sentence-transformers model.

A paired t-test found no statistically significant difference between strategies (p=0.298, 95% CI [-0.018, 0.056]). Question-level analysis shows hybrid outperforms on treatment/prevention queries while dense performs better on symptom queries — suggesting optimal strategy selection is query-dependent.

See `notebooks/exploration.ipynb` for full visualizations.

---

## Safety Layer

The system intercepts three categories of queries before they reach the LLM:

- **Emergency:** chest pain, stroke symptoms, breathing difficulty, overdose, suicidal ideation — returns hardcoded 911 directive
- **Dosage:** medication dosage requests — returns pharmacist referral
- **Diagnosis:** "do I have X" requests — returns healthcare provider referral

A confidence gate additionally blocks responses when retrieval similarity falls below 0.45 — the system refuses to answer rather than generate a low-confidence response.

---

*Built by — Mohit Sharma, MS Data Science, Northeastern University*