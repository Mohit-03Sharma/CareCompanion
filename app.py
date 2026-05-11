import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CareCompanion",
    layout="centered"
)

st.title("CareCompanion")
st.caption("Evidence-grounded health assistant powered by NIH MedlinePlus")

with st.sidebar:
    st.header("Settings")
    strategy = st.radio(
        "Retrieval Strategy",
        ["hybrid", "dense"],
        help="Hybrid combines semantic + keyword search. Dense uses semantic search only."
    )

    st.divider()
    st.header("System Stats")

    try:
        drift = requests.get(f"{API_URL}/drift-status", timeout=3).json()
        st.metric("Total Queries", drift.get("total_queries", 0))
        st.metric("Mean Confidence", drift.get("mean_confidence", 0))
        st.metric("Safety Flags", drift.get("safety_flags", 0))
    except Exception:
        st.warning("API not reachable")

    st.divider()
    st.header("A/B Results")

    try:
        exp = requests.get(f"{API_URL}/experiment-results", timeout=3).json()
        for row in exp.get("results", []):
            st.metric(
                f"{row['strategy'].title()} Strategy",
                f"{row['mean_score']:.4f}"
            )
    except Exception:
        st.warning("No experiment data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['topic']}]({s['url']})")
        if msg.get("confidence") is not None:
            confidence = msg["confidence"]
            color = "green" if confidence > 0.5 else "orange" if confidence > 0.3 else "red"
            st.caption(f"Confidence: :{color}[{confidence:.2f}]")
        if msg.get("safety_flagged"):
            st.warning(f"Safety flag: {msg.get('safety_category', 'flagged')}")

if query := st.chat_input("Ask a health question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge base..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": query, "strategy": strategy},
                    timeout=30
                ).json()

                answer = response["answer"]
                sources = response.get("sources", [])
                confidence = response.get("confidence", 0)
                safety_flagged = response.get("safety_flagged", False)
                safety_category = response.get("safety_category")

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- [{s['topic']}]({s['url']})")

                color = "green" if confidence > 0.5 else "orange" if confidence > 0.3 else "red"
                st.caption(f"Confidence: :{color}[{confidence:.2f}]")

                if safety_flagged:
                    st.warning(f"Safety flag: {safety_category}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "safety_flagged": safety_flagged,
                    "safety_category": safety_category
                })

            except Exception as e:
                st.error(f"Error: {e}")