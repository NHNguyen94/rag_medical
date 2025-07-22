import os

import streamlit as st

from src.clients.admin_client import AdminClient
from src.ui.utils import login_or_signup
from src.utils.enums import AdminConfig


def main_app():
    st.set_page_config(page_title="Update System Prompt", page_icon="🛠️")

    st.title("🛠️ Update System Prompt")

    admin_client = AdminClient(base_url=os.getenv("API_URL"), api_version="v1")

    user_id = st.session_state.get("hashed_username", "default_user_id")

    sys_prompt_dir = AdminConfig.CUSTOMIZED_SYSTEM_PROMPT_DIR

    with st.form("system_prompt_form"):
        system_prompt = st.text_area("System Prompt", height=150)
        reasoning_effort = st.selectbox("Reasoning Effort", ["low", "medium", "high"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        similarity_top_k = st.number_input("Similarity Top K", min_value=1, max_value=100, value=5)
        yml_file = f"{user_id}_system_prompt.yml"
        yml_path = os.path.join(sys_prompt_dir, yml_file)

        submitted = st.form_submit_button("Update Prompt")

        if submitted:
            try:
                response = admin_client.update_system_prompt(
                    system_prompt=system_prompt,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    similarity_top_k=similarity_top_k,
                    yml_file=yml_path,
                )
                st.success("✅ Prompt updated successfully!")
                st.json(response)
            except Exception as e:
                st.error(f"❌ Failed to update system prompt: {e}")


def run():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
