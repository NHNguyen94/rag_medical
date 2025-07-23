import os

import streamlit as st

from src.clients.history_client import HistoryClient
from src.ui.utils import login_or_signup


def chat_history_app():
    st.set_page_config(page_title="Chat History", page_icon="🧹")

    st.title("🧹 Manage Chat History")

    history_client = HistoryClient(base_url=os.getenv("API_URL"), api_version="v1")

    user_id = st.session_state.get("hashed_username", "default_user_id")

    st.markdown("Click the button below to **permanently delete** your chat history.")

    if st.button("🧹 Clear Chat History"):
        try:
            response = history_client.delete_chat_history(user_id=user_id)
            st.success("✅ Chat history cleared successfully!")
            st.json(response)
        except Exception as e:
            st.error(f"❌ Failed to delete chat history: {e}")


def run():
    if st.session_state.get("authenticated"):
        chat_history_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
