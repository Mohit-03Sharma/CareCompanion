import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are CareCompanion, a patient health assistant.
Your job is to answer health questions using ONLY the medical information provided to you.

Rules:
- Answer only from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say: "I don't have reliable information on this. Please consult a healthcare provider."
- Always mention which topic your answer comes from.
- Never diagnose. Never recommend specific medications or dosages.
- If the question involves emergency symptoms, tell the user to call 911 immediately."""


def format_context(retrieved_chunks):
    """Turn retrieved chunks into a readable context block for the LLM."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i + 1} — {chunk['topic']} ({chunk['url']})]\n{chunk['content']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(query, retrieved_chunks):
    """
    Send the query + retrieved chunks to Llama 3.1 via Groq.
    Returns the answer text and the sources used.
    """
    context = format_context(retrieved_chunks)

    user_message = f"""Context:
{context}

Question: {query}

Answer using only the context above."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=512
    )

    answer = response.choices[0].message.content

    sources = list({
        chunk["url"]: {"topic": chunk["topic"], "url": chunk["url"]}
        for chunk in retrieved_chunks
    }.values())

    return answer, sources
