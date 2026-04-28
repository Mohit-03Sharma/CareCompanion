import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "dbname": os.getenv("DB_NAME", "carecompanion"),
    "user": os.getenv("DB_USER", "carecompanion"),
    "password": os.getenv("DB_PASSWORD", "carecompanion"),
}


def get_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def setup_database():
    """
    Create all tables and enable pgvector extension.
    Run this once to initialize the database schema.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Enable pgvector extension — this is what allows us to store
    # and search embeddings (384-dimensional vectors) directly in Postgres
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # knowledge_chunks: stores every parsed chunk of medical text
    # along with its embedding vector for semantic search
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id SERIAL PRIMARY KEY,
            source VARCHAR(50) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            url TEXT,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create an index on the embedding column using ivfflat algorithm
    # This makes vector similarity search fast at scale
    # probes=10 means check 10 clusters during search — balances speed vs accuracy
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx
        ON knowledge_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    # query_logs: stores every user query with metadata
    # This feeds our drift monitoring in Week 5
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            query_embedding vector(384),
            retrieval_strategy VARCHAR(20),
            confidence_score FLOAT,
            safety_flagged BOOLEAN DEFAULT FALSE,
            safety_category VARCHAR(50),
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # experiment_runs: stores A/B experiment results for Week 4
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id SERIAL PRIMARY KEY,
            experiment_name VARCHAR(100) NOT NULL,
            strategy VARCHAR(20) NOT NULL,
            query TEXT NOT NULL,
            retrieved_chunks TEXT,
            relevance_score FLOAT,
            confidence_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database setup complete. All tables created.")


if __name__ == "__main__":
    setup_database()