import os
import requests
import streamlit as st


st.set_page_config(page_title="PDF RAG Chat", layout="centered")
st.title("PDF RAG Chat")

default_api_url = os.getenv("RAG_API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input("API base URL", value=default_api_url)

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("Clear chat"):
        st.session_state.messages = []
with col2:
    st.caption("Tip: start API first")


def _call_api(question_text: str) -> str:
    resp = requests.post(
        f"{api_url.rstrip('/')}/ask",
        json={"question": question_text},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data.get("answer", "")


# Render existing chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


user_msg = st.chat_input("Type your question and press Enter…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = _call_api(user_msg)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            err = str(e)
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})

