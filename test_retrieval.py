import sys
sys.path.insert(0, 'src')
from retrieval import retrieve_dense, retrieve_hybrid, compute_confidence

query = "What are the symptoms of diabetes?"

print("=== STRATEGY A: DENSE ===")
dense = retrieve_dense(query)
for r in dense:
    print(f"  [{r['similarity']:.3f}] {r['topic']} — {r['content'][:80]}")
print(f"Confidence: {compute_confidence(dense, 'dense')}")

print()
print("=== STRATEGY B: HYBRID ===")
hybrid = retrieve_hybrid(query)
for r in hybrid:
    print(f"  [{r['rrf_score']:.4f}] {r['topic']} — {r['content'][:80]}")
print(f"Confidence: {compute_confidence(hybrid, 'hybrid')}")