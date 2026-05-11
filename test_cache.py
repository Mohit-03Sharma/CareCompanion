import sys
import time
sys.path.insert(0, 'src')
from pipeline import run_pipeline

query = "What are the symptoms of diabetes?"

print("=== FIRST CALL (no cache) ===")
start = time.time()
result = run_pipeline(query)
elapsed = time.time() - start
print(f"Cache hit: {result.get('cache_hit')}")
print(f"Time: {elapsed:.2f}s")
print(f"Answer preview: {result['answer'][:100]}")

print()
print("=== SECOND CALL (should hit cache) ===")
start = time.time()
result = run_pipeline(query)
elapsed = time.time() - start
print(f"Cache hit: {result.get('cache_hit')}")
print(f"Time: {elapsed:.2f}s")
print(f"Answer preview: {result['answer'][:100]}")

print()
print("=== THIRD CALL — semantically similar query ===")
similar_query = "What symptoms does diabetes cause?"
start = time.time()
result = run_pipeline(similar_query)
elapsed = time.time() - start
print(f"Cache hit: {result.get('cache_hit')}")
print(f"Cache similarity: {result.get('cache_similarity')}")
print(f"Time: {elapsed:.2f}s")