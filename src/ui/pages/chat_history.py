import os
import pandas as pd
import io
import base64
import streamlit as st

from fpdf import FPDF
from datetime import datetime

from src.clients.history_client import HistoryClient, ChatPDF, ChatExportManager
from src.ui.utils import login_or_signup


def chat_history_app():
    st.set_page_config(page_title="Chat History", page_icon="🧹")

    st.title("🧹 Manage Chat History")

    history_client = HistoryClient(base_url=os.getenv("API_URL"), api_version="v1")
    export_manager = ChatExportManager()

    user_id = st.session_state.get("hashed_username", "default_user_id")

    st.markdown("Click the button below to **permanently delete** your chat history.")

    if st.button("🧹 Clear Chat History"):
        try:
            response = history_client.delete_chat_history(user_id=user_id)
            st.success("✅ Chat history cleared successfully!")
            st.json(response)
        except Exception as e:
            st.error(f"❌ Failed to delete chat history: {e}")

    try:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            limit = st.selectbox("Select number of messages to display", [5, 10, 20, 50], index=1)

        history = history_client.get_chat_history(user_id=user_id, limit=limit)
        if history:
            df = export_manager.generate_chat_dataframe(history)

            with col2:
                excel_data = export_manager.export_to_excel(df)
                st.download_button(
                    label="📄 Export to Excel",
                    data=excel_data,
                    file_name="chat_history.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col3:
                pdf_data = export_manager.export_to_pdf(df)
                st.download_button(
                    label="📑 Export to PDF",
                    data=pdf_data,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )

            for idx, chat in enumerate(history):
                with st.container():
                    col1, col2 = st.columns([0.95, 0.05])

                    with col1:
                        with st.chat_message("user"):
                            st.markdown(chat["message"])
                            dt = datetime.fromisoformat(chat["created_at"])
                            formatted_time = dt.strftime("%B %d, %Y at %H:%M")
                            st.caption(f"🕒 {formatted_time}")

                        with st.chat_message("assistant"):
                            st.markdown(chat["response"])

                    with col2:
                        delete_button = st.button("🗑️", key=f"delete_{chat['id']}", help="Delete this chat")
                        if delete_button:
                            try:
                                history_client.delete_single_chat_message(chat_id=chat["id"])
                                st.success("Deleted chat successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
        else:
            st.info("No chat history found.")
    except Exception as e:
        st.error(f"Failed to fetch chat history: {e}")

def run():
    if st.session_state.get("authenticated"):
        chat_history_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
