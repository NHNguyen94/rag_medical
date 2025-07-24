import os
from typing import Optional

import dotenv
import streamlit as st

from src.clients.auth_client import AuthClient
from src.clients.chat_client import ChatClient
from src.utils.enums import AudioConfig, AdminConfig
from src.utils.helpers import hash_string, get_unique_id

import requests

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
    chat_client: ChatClient, 
    user_id: str, 
    message: str, 
    selected_domain: str, 
    disable_emotion_recognition: bool = False,
    selected_model: Optional[str] = None,
    customized_sys_prompt_path: Optional[str] = None,
    bypass_cache: bool = False,
    language: str = "English",
):
    try:
        response_data = chat_client.chat(
          user_id=user_id,
          message=message,
          selected_domain=selected_domain,
          disable_emotion_recognition=disable_emotion_recognition,
          model_name=selected_model,
          customized_sys_prompt_path=customized_sys_prompt_path,
          bypass_cache=bypass_cache,
          language=language,
          use_qr=st.session_state.enable_recommendation
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
    model_name: Optional[str] = None,  # New parameter
):
    try:
        response_data = chat_client.chat(
            user_id=user_id,
            message=message,
            selected_domain=selected_domain,
            customized_sys_prompt_path=customized_sys_prompt_path,
            customize_index_path=customize_index_path,
            model_name=model_name,  # Pass to chat client
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

        if "messages" not in st.session_state or not st.session_state.messages:
            st.session_state.messages = [
          {
              "role": "assistant",
              "content": "Hello, I'm your AI medical assistant. How can I help you today?",
          }
      ]

        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    feedback_key = f"feedback_{hash_string(message['content'])}_{idx}"
                    feedback_given = st.session_state.get(feedback_key, None)
                    if feedback_given is None:
                        col1, col2, col3 = st.columns([1, 1, 3])
                        like_clicked = col1.button("👍", key=f"like_{hash_string(message['content'])}_{idx}")
                        dislike_clicked = col2.button("👎", key=f"dislike_{hash_string(message['content'])}_{idx}")
                        if like_clicked or dislike_clicked:
                            feedback_type = "like" if like_clicked else "dislike"
                            st.session_state[feedback_key] = feedback_type
                            send_feedback_to_backend(user_id, message["content"], message["content"], feedback_type)
                            if like_clicked:
                                st.success("Thank you for your feedback!")
                            else:
                                st.warning("Sorry this answer did not help you.")
                    elif feedback_given == "like":
                        st.success("Thank you for your feedback!")
                    elif feedback_given == "dislike":
                        st.warning("Sorry this answer did not help you.")
                        if st.button("Try Again", key=f"regen_{hash_string(message['content'])}_{idx}"):
                            with st.spinner("Generating a new answer..."):
                                # Find the previous user message before this assistant message
                                user_message = None
                                for prev in reversed(st.session_state.messages[:idx]):
                                    if prev["role"] == "user":
                                        user_message = prev["content"]
                                        break
                                if user_message:
                                    # Re-run the chat with the same user message
                                    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
                                    handle_chat_response_with_voice(
                                        chat_client,
                                        user_id,
                                        user_message,
                                        selected_domain,
                                        customized_sys_prompt_path,
                                        customize_index_path,
                                    )
                                    st.rerun()

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


def send_feedback_to_backend(user_id, message, response, feedback_type):
    api_url = os.getenv("API_URL", "http://localhost:8000")  # Adjust as needed
    endpoint = f"{api_url}/v1/chatbot/feedback"
    payload = {
        "user_id": user_id,
        "message": message,
        "response": response,
        "feedback_type": feedback_type,
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=5)
        r.raise_for_status()
        return r.json().get("success", False)
    except Exception as e:
        print(f"Error sending feedback: {e}")
        return False
