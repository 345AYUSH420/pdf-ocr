import os
import requests
import streamlit as st

st.set_page_config(page_title="PDF RAG Chat", layout="centered")
st.title("PDF RAG Chat")

default_api_url = os.getenv("RAG_API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input("API base URL", value=default_api_url)

# -------------------------------
# 🔹 SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = 1

if "pending_options" not in st.session_state:
    st.session_state.pending_options = []


# -------------------------------
# 🔹 CLEAR CHAT
# -------------------------------
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.step = 1
    st.session_state.pending_options = []


# -------------------------------
# 🔹 API CALL
# -------------------------------
def _call_api(question_text: str, history: list, step: int) -> str:
    resp = requests.post(
        f"{api_url.rstrip('/')}/ask",
        json={
            "question": question_text,
            "history": history,
            "step": step   # 🔥 IMPORTANT
        },
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    return resp.json().get("answer", "")


# -------------------------------
# 🔹 PARSE OPTIONS
# -------------------------------
def parse_options(text):
    options = []
    for line in text.split("\n"):
        if "☐" in line:
            options.append(line.replace("☐", "").strip())
    return options


# -------------------------------
# 🔹 RENDER CHAT
# -------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# -------------------------------
# 🔹 USER INPUT (only if no options)
# -------------------------------
if not st.session_state.pending_options:

    user_msg = st.chat_input("Type your question...")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = _call_api(
                        user_msg,
                        st.session_state.messages,
                        st.session_state.step
                    )

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                options = parse_options(answer)

                # 🔥 IF OPTIONS → diagnosis mode
                if options:
                    st.session_state.pending_options = options
                    st.session_state.step += 1
                else:
                    # 🔥 QA MODE → reset step
                    st.session_state.step = 1

            except Exception as e:
                st.error(str(e))


# -------------------------------
# 🔹 OPTION UI (checkbox / radio)
# -------------------------------
if st.session_state.pending_options:

    st.markdown("### Select your answer:")

    options = st.session_state.pending_options
    selected_values = []

    # YES/NO → RADIO
    if len(options) == 2 and set(o.lower() for o in options) == {"yes", "no"}:
        selected = st.radio("Choose one:", options)
        if selected:
            selected_values = [selected]

    # MULTI → CHECKBOX
    else:
        for i, opt in enumerate(options):
            if st.checkbox(opt, key=f"chk_{st.session_state.step}_{i}"):
                selected_values.append(opt)

    # SUBMIT
    if st.button("Submit Selection"):

        if not selected_values:
            st.warning("Select at least one option")
        else:
            user_response = ", ".join(selected_values)

            st.session_state.messages.append({
                "role": "user",
                "content": user_response
            })

            with st.chat_message("user"):
                st.write(user_response)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        answer = _call_api(
                            user_response,
                            st.session_state.messages,
                            st.session_state.step
                        )

                    st.write(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    options = parse_options(answer)

                    if options:
                        st.session_state.pending_options = options
                        st.session_state.step += 1
                    else:
                        st.session_state.pending_options = []
                        st.session_state.step = 1  # reset after flow

                except Exception as e:
                    st.error(str(e))