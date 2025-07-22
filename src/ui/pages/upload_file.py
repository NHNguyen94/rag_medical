import os

import streamlit as st

from src.clients.admin_client import AdminClient
from src.ui.utils import (
    login_or_signup,
    generate_customized_csv_file_path,
)
from src.utils.directory_manager import DirectoryManager
from src.utils.enums import AdminConfig, IngestionConfig

directory_manager = DirectoryManager()


def main_app():
    st.set_page_config(page_title="Upload CSV and Ingest", page_icon="📥")

    st.title("📥 Upload CSV File for Vector Store Ingestion")

    admin_client = AdminClient(base_url=os.getenv("API_URL"), api_version="v1")

    user_id = st.session_state.get("hashed_username", "default_user_id")

    col_name_to_ingest = IngestionConfig.COL_NAME_TO_INGEST
    uploaded_file = st.file_uploader(
        f"Upload a CSV file with data in column {col_name_to_ingest}", type=["csv"]
    )

    # Create directories if they do not exist, clear old data
    customized_csv_dir = f"{AdminConfig.CUSTOMIZED_CSV_DIR}/{user_id}"
    if directory_manager.check_if_file_exists(customized_csv_dir):
        directory_manager.delete_non_empty_dir(customized_csv_dir)

    directory_manager.create_dir_if_not_exists(customized_csv_dir)

    uploaded_file_path = os.path.join(
        customized_csv_dir, generate_customized_csv_file_path(user_id)
    )

    if uploaded_file is not None:
        st.write("✅ File uploaded:", uploaded_file.name)

        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"📂 File saved to: {uploaded_file_path}")

        if st.button("Ingest File"):
            try:
                # delete all indexes to load new one
                cutomized_index_dir = f"{AdminConfig.CUSTOMIZED_INDEX_DIR}/{user_id}"
                if directory_manager.check_if_dir_exists(cutomized_index_dir):
                    directory_manager.delete_non_empty_dir(cutomized_index_dir)

                response = admin_client.ingest_custom_file(
                    file_path=customized_csv_dir,
                    index_path=cutomized_index_dir,
                )
                st.success("🚀 File ingested successfully!")
                st.json(response)
            except Exception as e:
                st.error(f"❌ Failed to ingest file: {e}")
    else:
        st.info("📎 Please upload a CSV file to continue.")


def run():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
