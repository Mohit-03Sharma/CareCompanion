import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.header("Clinical Note Extractor")
st.caption("Extract structured information from unstructured clinical notes")

st.info(
    "In production, this page would require clinician authentication. "
    "See README for architecture notes on role-based access control."
)

# Input method toggle
input_method = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True)

note_text = None
uploaded_file = None

if input_method == "Paste text":
    note_text = st.text_area(
        "Paste clinical note here",
        height=300,
        placeholder="Paste discharge summary, clinical note, or patient record..."
    )
else:
    uploaded_file = st.file_uploader("Upload clinical note PDF", type=["pdf"])

extract_button = st.button("Extract Information", type="primary")

if extract_button:
    if not note_text and not uploaded_file:
        st.error("Please provide a clinical note — either paste text or upload a PDF.")
    else:
        with st.spinner("Extracting clinical information..."):
            try:
                if uploaded_file:
                    response = requests.post(
                        f"{API_URL}/extract-note",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        timeout=30
                    )
                else:
                    response = requests.post(
                        f"{API_URL}/extract-note",
                        data={"text": note_text},
                        timeout=30
                    )

                result = response.json()

                if "error" in result:
                    st.error(f"Extraction failed: {result['error']}")
                else:
                    # Patient overview
                    st.subheader("Patient Overview")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Age", result.get("patient_age", "N/A"))
                    col2.metric("Gender", result.get("patient_gender", "N/A"))
                    col3.metric("Admission", result.get("admission_date", "N/A"))

                    st.markdown(f"**Chief Complaint:** {result.get('chief_complaint', 'N/A')}")

                    # Diagnoses
                    st.subheader("Diagnoses")
                    for d in result.get("diagnoses", []):
                        st.markdown(f"- {d}")

                    # Medications
                    st.subheader("Discharge Medications")
                    meds = result.get("discharge_medications", [])
                    if meds:
                        for m in meds:
                            is_new = m.get("is_new", False)
                            badge = " [NEW]" if is_new else ""
                            st.markdown(f"- **{m.get('name')}** {m.get('dose', '')}{badge}")

                    # Follow-up
                    st.subheader("Follow-up Actions")
                    for f in result.get("follow_up_actions", []):
                        st.markdown(f"- {f}")

                    # Warning signs
                    st.subheader("Warning Signs — Return to ED If:")
                    warnings = result.get("warning_signs", [])
                    if warnings:
                        for w in warnings:
                            st.warning(w)

                    # Dietary instructions
                    dietary = result.get("dietary_instructions", [])
                    if dietary:
                        st.subheader("Dietary Instructions")
                        for d in dietary:
                            st.markdown(f"- {d}")

                    # Validation warnings
                    st.subheader("Knowledge Base Validation")
                    val_warnings = result.get("validation_warnings", [])
                    if val_warnings:
                        for w in val_warnings:
                            st.error(f"Not in knowledge base: {w}")
                    else:
                        st.success("All diagnoses validated against NIH knowledge base")

                    # Raw JSON download
                    st.subheader("Download Structured Output")
                    st.download_button(
                        label="Download as JSON",
                        data=json.dumps(result, indent=2),
                        file_name="extracted_note.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Error: {e}")