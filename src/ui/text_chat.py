import os

import streamlit as st
import torch

from src.clients.chat_client import ChatClient
from src.ui.utils import login_or_signup, handle_chat_response
from src.utils.enums import ChatBotConfig
from src.utils.helpers import clean_document_text

torch.classes.__path__ = []


def main_app():
    prompt = None
    st.title("📝 Text Chat - AI Medical Assistant")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

    selected_domain = st.selectbox("Select a medical domain", ChatBotConfig.DOMAINS)
    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
    user_id = st.session_state.get("hashed_username", "default_user_id")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I'm your AI medical assistant. How can I help you today?"}]
    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []
    if "retrieved_documents" not in st.session_state:
        st.session_state.retrieved_documents = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.markdown("### 💡 What's on your mind?")
        with st.expander("Need inspiration? Pick a topic and get AI-generated questions!"):
            topic = st.selectbox("Choose a topic", ChatBotConfig.DOMAINS, key="topic_picker")
            if st.button("✨ Generate Questions"):
                with st.spinner("Generating questions..."):
                    topic_response = chat_client.get_ai_question(user_id=user_id, topic=topic)
                    if topic_response and "recommended_question" in topic_response:
                        st.session_state.prefilled_prompt = topic_response["recommended_question"] # 👈 Set prefill
                        st.session_state.show_prefilled = True
                        st.rerun()

    if st.session_state.get("show_prefilled"):
        if "submitted_prefilled" not in st.session_state:
            st.session_state.submitted_prefilled = False

        if not st.session_state.submitted_prefilled:
            prompt = st.text_input(
                "✏️ Edit or send the AI-generated question:",
                value=st.session_state.prefilled_prompt,
                key="prefilled_input"
            )
            if st.button("Send", key="send_prefilled_btn") and not st.session_state.submitted_prefilled:
                st.session_state.submitted_prefilled = True
                st.session_state.prefilled_prompt = prompt
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.handle_prefilled_now = True
                st.rerun()
        else:
            if st.session_state.get("handle_prefilled_now"):
                handle_chat_response(
                    chat_client,
                    user_id,
                    st.session_state.prefilled_prompt,
                    selected_domain
                )
                # Reset state
                st.session_state.show_prefilled = False
                st.session_state.handle_prefilled_now = False
                st.session_state.prefilled_prompt = ""
                st.session_state.submitted_prefilled = False

    prompt = st.chat_input("Type your message here...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        handle_chat_response(chat_client, user_id, prompt, selected_domain)

    if st.session_state.followup_questions:
        st.divider()
        st.markdown("Related")
        for idx, q in enumerate(st.session_state.followup_questions):
            if st.button(f"➕ {q}", key=f"followup_{idx}"):
                st.session_state.messages.append({"role": "user", "content": q})
                if handle_chat_response(chat_client, user_id, q, selected_domain):
                    st.rerun()

    with st.sidebar:
        st.header("Retrieved Documents")
        if st.session_state.retrieved_documents:
            for i, doc in enumerate(st.session_state.retrieved_documents):
                st.markdown(f"**Doc {i + 1}:** {clean_document_text(doc)}", unsafe_allow_html=True)
        else:
            st.markdown("_No documents retrieved yet._")


def run():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
