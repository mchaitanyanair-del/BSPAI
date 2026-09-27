"""
Streamlit UI. Run with:
    streamlit run frontend/streamlit_app.py
Assumes the FastAPI backend is running at BACKEND_URL.
"""
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="BiSpec Pairwise AI", layout="wide")
st.title("BiSpec Pairwise AI — Target Pair Explorer")

tab1, tab2 = st.tabs(["Pair Report", "Ask a question"])

with tab1:
    col1, col2 = st.columns(2)
    gene_a = col1.text_input("Gene 1", "BCMA")
    gene_b = col2.text_input("Gene 2", "CD3")

    if st.button("Generate report"):
        with st.spinner("Running model + retrieval..."):
            resp = requests.post(f"{BACKEND_URL}/report",
                                  json={"gene_a": gene_a, "gene_b": gene_b})
        if resp.ok:
            data = resp.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Model score", data["model_score"])
            c2.metric("EB prior score", data["eb_score"])
            c3.metric("Coverage", data["coverage"])

            st.subheader("Explanation")
            st.write(data["explanation"])

            with st.expander("Raw structured features"):
                st.json(data)
        else:
            st.error(resp.text)

with tab2:
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Ask about a gene pair or the method...")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        resp = requests.post(f"{BACKEND_URL}/chat", json={
            "session_id": st.session_state.session_id,
            "message": user_msg,
        })
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        reply = data["reply"]

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)
