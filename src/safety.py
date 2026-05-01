import re

# Emergency patterns which should bypass the LLM entirely
# A wrong or delayed answer here could cost someone their life
EMERGENCY_PATTERNS = [
    r"chest pain",
    r"heart attack",
    r"can't breathe",
    r"cannot breathe",
    r"difficulty breathing",
    r"stroke",
    r"unconscious",
    r"not breathing",
    r"severe bleeding",
    r"overdose",
    r"suicide",
    r"kill myself",
    r"want to die",
    r"poisoning",
    r"seizure",
]

# Out of scope patterns, things no health assistant should answer
OUT_OF_SCOPE_PATTERNS = [
    r"what (dose|dosage|mg|milligram)",
    r"how much (medication|medicine|drug|pill|tablet)",
    r"should i take",
    r"can i take .* with",
    r"diagnose me",
    r"do i have",
    r"prescribe",
]

# Safe responses for each category - hardcoded, never LLM-generated
EMERGENCY_RESPONSE = (
    "This sounds like a medical emergency. "
    "Please call 911 immediately or go to your nearest emergency room. "
    "Do not wait — get help now."
)

DOSAGE_RESPONSE = (
    "I'm not able to provide medication dosage advice. "
    "Please consult your doctor or pharmacist for guidance on medications."
)

DIAGNOSIS_RESPONSE = (
    "I'm not able to diagnose medical conditions. "
    "Please see a healthcare provider for a proper evaluation."
)

# Confidence threshold — below this we don't trust the retrieval
# Tuned based on the distribution of similarity scores we observed
CONFIDENCE_THRESHOLD = 0.45

LOW_CONFIDENCE_RESPONSE = (
    "I don't have reliable information on this topic in my knowledge base. "
    "Please consult a healthcare provider or visit medlineplus.gov for accurate information."
)


def check_safety(query):
    """
    Run the query through the safety layer before hitting the LLM.
    
    Returns a tuple: (is_safe, category, response)
    - is_safe: False means return the hardcoded response, skip the LLM
    - category: what type of flag was triggered
    - response: the hardcoded safe response to return
    """
    query_lower = query.lower()

    # Layer 1 — Emergency check
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, query_lower):
            return False, "emergency", EMERGENCY_RESPONSE

    # Layer 2 — Dosage check
    for pattern in OUT_OF_SCOPE_PATTERNS[:4]:
        if re.search(pattern, query_lower):
            return False, "dosage", DOSAGE_RESPONSE

    # Layer 3 — Diagnosis check
    for pattern in OUT_OF_SCOPE_PATTERNS[4:]:
        if re.search(pattern, query_lower):
            return False, "diagnosis", DIAGNOSIS_RESPONSE

    return True, None, None


def check_confidence(confidence_score):
    """
    Gate on retrieval confidence.
    If retrieved chunks aren't similar enough to the query,
    we don't trust the LLM answer they would produce.
    """
    if confidence_score < CONFIDENCE_THRESHOLD:
        return False, LOW_CONFIDENCE_RESPONSE
    return True, None