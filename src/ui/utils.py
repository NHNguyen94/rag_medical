import os
from typing import Optional

import dotenv
import streamlit as st

from src.clients.auth_client import AuthClient
from src.clients.chat_client import ChatClient
from src.utils.enums import AudioConfig, AdminConfig
from src.utils.helpers import hash_string, get_unique_id

dotenv.load_dotenv()


def login_or_signup():
    st.title("Login / Sign Up")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "is_signup" not in st.session_state:
        st.session_state.is_signup = False

    if st.session_state.get("just_registered"):
        st.session_state.is_signup = False
        del st.session_state["just_registered"]

    auth_client = AuthClient(base_url=os.getenv("API_URL"), api_version="v1")

    is_signup = st.checkbox("Create a new account?", value=st.session_state.is_signup)
    if is_signup != st.session_state.is_signup:
        st.session_state.is_signup = is_signup

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.session_state.is_signup:
        confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Submit"):
        if st.session_state.is_signup:
            if password != confirm_password:
                st.error("Passwords do not match.")
                return
            try:
                auth_client.register(username, password)
                st.success("Account created successfully! You can now log in.")
                st.session_state["just_registered"] = True
                st.session_state.is_signup = False
                st.rerun()
            except Exception as e:
                st.error(f"Signup failed: {e}")
        else:
            try:
                auth_client.login(username, password)
                st.session_state.authenticated = True
                st.session_state.hashed_username = hash_string(username)
                st.success("Login successful!")
                st.rerun()
            except Exception as e:
                st.error("Login failed, username or password is incorrect")


def handle_chat_response(
    chat_client: ChatClient, user_id: str, message: str, selected_domain: str
):
    try:
        response_data = chat_client.chat(
            user_id=user_id,
            message=message,
            selected_domain=selected_domain,
            use_qr=st.session_state.enable_recommendation,
        )
        st.session_state.retrieved_documents = response_data.get(
            "nearest_documents", []
        )
        st.session_state.followup_questions = response_data.get(
            "recommended_questions", []
        )
        response = response_data.get("response", "No response from the assistant.")
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return True
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"An error occurred: {e}")
        return False


def handle_chat_response_with_voice(
    chat_client: ChatClient,
    user_id: str,
    message: str,
    selected_domain: str,
    customized_sys_prompt_path: Optional[str] = None,
    customize_index_path: Optional[str] = None,
):
    try:
        response_data = chat_client.chat(
            user_id=user_id,
            message=message,
            selected_domain=selected_domain,
            customized_sys_prompt_path=customized_sys_prompt_path,
            customize_index_path=customize_index_path,
        )
        st.session_state.retrieved_documents = response_data.get(
            "nearest_documents", []
        )
        st.session_state.followup_questions = response_data.get(
            "recommended_questions", []
        )
        response = response_data.get("response", "No response from the assistant.")

        output_tts_dir = AudioConfig.OUTPUT_TTS_AUDIO_DIR
        output_tts_file_name = f"{user_id}_tts_{get_unique_id()}.wav"
        output_tts_path = os.path.join(output_tts_dir, output_tts_file_name)
        chat_client.text_to_speech(
            text=response,
            audio_path=output_tts_path,
        )

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.audio(output_tts_path, format=AudioConfig.AUDIO_FORMAT)
        # st.session_state.audio_ready = False

        return True
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"An error occurred: {e}")
        return False


def define_customized_sys_prompt_path(user_id: str) -> str:
    return f"{user_id}_system_prompt.yml"


def generate_customized_csv_file_path(user_id: str) -> str:
    return f"{user_id}_customized_file_{str(get_unique_id())}.csv"


def define_customized_index_file_path(user_id: str) -> str:
    return f"{AdminConfig.CUSTOMIZED_INDEX_DIR}/{user_id}"
