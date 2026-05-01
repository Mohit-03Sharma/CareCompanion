import sys
sys.path.insert(0, 'src')
from retrieval import retrieve_hybrid, compute_confidence
from llm import generate_answer

query = "What are the early symptoms of diabetes?"

print(f"Query: {query}")
print("Retrieving chunks...")
chunks = retrieve_hybrid(query)
confidence = compute_confidence(chunks, "hybrid")
print(f"Confidence: {confidence}")
print()

print("Generating answer...")
answer, sources = generate_answer(query, chunks)

print("=== ANSWER ===")
print(answer)
print()
print("=== SOURCES ===")
for s in sources:
    print(f"  {s['topic']} — {s['url']}")