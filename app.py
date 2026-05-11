import streamlit as st

st.set_page_config(page_title="CareCompanion", layout="centered")

st.title("CareCompanion")
st.caption("Clinical intelligence platform — patient Q&A and clinician note extraction")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Assistant")
    st.write("Evidence-grounded health Q&A powered by NIH MedlinePlus content.")
    st.page_link("pages/patient.py", label="Open Patient Assistant")

with col2:
    st.subheader("Clinician Extractor")
    st.write("Extract structured information from unstructured clinical notes.")
    st.page_link("pages/clinician.py", label="Open Clinical Extractor")

st.divider()
st.caption(
    "Note: Production deployment would include role-based authentication "
    "separating patient and clinician access. See README for architecture details."
)
