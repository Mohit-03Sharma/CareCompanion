import json
import pdfplumber
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

EXTRACTION_PROMPT = """You are a clinical information extraction system.
Extract the following information from the clinical note provided.
Respond ONLY with a valid JSON object — no preamble, no explanation, no markdown.

Extract these fields:
- patient_age: integer or null
- patient_gender: string or null
- admission_date: string or null
- discharge_date: string or null
- chief_complaint: string — main reason for admission
- diagnoses: list of strings — all diagnoses mentioned
- medications_on_admission: list of objects with name and dose
- discharge_medications: list of objects with name, dose, and is_new (boolean)
- follow_up_actions: list of strings — all follow-up appointments and timelines
- warning_signs: list of strings — symptoms that require immediate return to ED
- dietary_instructions: list of strings or empty list
- red_flags: list of strings — anything clinically concerning in this note

Return only the JSON object."""


def extract_text_from_pdf(file_path):
    """Extract raw text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_txt(file_path):
    """Read plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_from_text(note_text):
    """
    Run extraction pipeline on raw clinical note text.
    Returns structured dict with all extracted fields.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"{EXTRACTION_PROMPT}\n\nCLINICAL NOTE:\n{note_text}"
            }
        ],
        temperature=0.1,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def validate_diagnoses(diagnoses, knowledge_base_fn):
    """
    Cross-reference extracted diagnoses against NIH knowledge base.
    Returns list of warnings for diagnoses not found in knowledge base.
    """
    warnings = []
    known_topics = knowledge_base_fn()

    for diagnosis in diagnoses:
        diagnosis_lower = diagnosis.lower()
        matched = any(
            diagnosis_lower in topic.lower() or topic.lower() in diagnosis_lower
            for topic in known_topics
        )
        if not matched:
            warnings.append(
                f"'{diagnosis}' not found in knowledge base — "
                f"verify against clinical guidelines"
            )

    return warnings


def get_known_topics():
    """Fetch all distinct topics from the knowledge base."""
    from database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT topic FROM knowledge_chunks")
    topics = [row["topic"] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return topics


def process_note(text=None, file_path=None):
    """
    Main entry point. Accepts raw text or a file path (txt or pdf).
    Returns structured extraction with validation warnings.
    """
    if file_path:
        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_txt(file_path)

    if not text or len(text.strip()) < 100:
        return {"error": "Note text is too short or empty"}

    extracted = extract_from_text(text)
    warnings = validate_diagnoses(
        extracted.get("diagnoses", []),
        get_known_topics
    )
    extracted["validation_warnings"] = warnings

    return extracted
