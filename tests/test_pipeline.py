from safety import check_safety, check_confidence
from retrieval import chunk_text_helper


# Safety layer tests

def test_emergency_query_flagged():
    is_safe, category, _ = check_safety("I have chest pain and can't breathe")
    assert not is_safe
    assert category == "emergency"


def test_dosage_query_flagged():
    is_safe, category, _ = check_safety("What dosage of ibuprofen should I take?")
    assert not is_safe
    assert category == "dosage"


def test_diagnosis_query_flagged():
    is_safe, category, _ = check_safety("Do I have diabetes?")
    assert not is_safe
    assert category == "diagnosis"


def test_normal_query_passes_safety():
    is_safe, category, _ = check_safety("What are the symptoms of diabetes?")
    assert is_safe
    assert category is None


def test_low_confidence_gated():
    confident, response = check_confidence(0.1)
    assert not confident
    assert response is not None


def test_high_confidence_passes():
    confident, response = check_confidence(0.8)
    assert confident
    assert response is None


# --- Chunking tests ---

def test_chunk_produces_output():
    text = "a" * 1000
    chunks = chunk_text_helper(text, chunk_size=400, overlap=50)
    assert len(chunks) > 0


def test_chunk_overlap_works():
    text = "a" * 1000
    chunks = chunk_text_helper(text, chunk_size=400, overlap=50)
    # With overlap, consecutive chunks should share content
    assert len(chunks) > 1000 // 400
