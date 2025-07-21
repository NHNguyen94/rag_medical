import os
import wave

import dotenv
import streamlit as st
import torch
from st_audiorec import st_audiorec

from src.clients.auth_client import AuthClient
from src.clients.chat_client import ChatClient
from src.utils.date_time_manager import DateTimeManager
from src.utils.directory_manager import DirectoryManager
from src.utils.enums import ChatBotConfig
from src.utils.helpers import clean_document_text, hash_string

datetime_manager = DateTimeManager()
directory_manager = DirectoryManager()

dotenv.load_dotenv()
# https://github.com/datalab-to/marker/issues/442
torch.classes.__path__ = []

def handle_chat_response(
        chat_client: ChatClient,
        user_id: str,
        message: str,
        selected_domain: str
) -> bool:
    try:
        response_data = chat_client.chat(
            user_id=user_id, message=message, selected_domain=selected_domain
        )

        nearest_docs = response_data.get("nearest_documents", [])
        st.session_state.retrieved_documents = nearest_docs

        followup_questions = response_data.get("recommended_questions", [])
        st.session_state.followup_questions = followup_questions

        response = response_data.get("response", "No response from the assistant.")
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        return True

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"An error occurred: {e}")
        return False


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


def main_app():
    st.title("AI-powered medical assistant")

    col1, col2 = st.columns([7, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()

    domain_options = ChatBotConfig.DOMAINS
    selected_domain = st.selectbox("Select a medical domain", domain_options)

    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
    user_id = st.session_state.get("hashed_username", "default_user_id")
    default_welcome_message = (
        "Hello, I'm your AI medical assistant. How can I help you today?"
    )

    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []

    if "retrieved_documents" not in st.session_state:
        st.session_state.retrieved_documents = []

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": default_welcome_message}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    use_voice_input = st.toggle("🎙️ Enable voice input")

    prompt = None

    if use_voice_input:
        st.markdown("Click below to record your message:")
        audio_bytes = st_audiorec()

        if audio_bytes:
            audio_dir = "src/data/recordings_from_speaker"
            directory_manager.create_dir_if_not_exists(audio_dir)
            timestamp = datetime_manager.get_current_local_time_str()
            wav_filename = f"{user_id}_recording_{timestamp}.wav"
            wav_path = os.path.join(audio_dir, wav_filename)

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(audio_bytes)

            st.success(f"Audio saved: {wav_path}")
            st.audio(wav_path, format="audio/wav")

            response_from_transcribe = chat_client.voice_chat(
                user_id=user_id,
                audio_file=wav_path,
                selected_domain=selected_domain,
            )

            prompt = response_from_transcribe.get("response", "No response from the assistant.")

    else:
        prompt = st.chat_input("Type your message here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        handle_chat_response(chat_client, user_id, prompt, selected_domain)

    if st.session_state.followup_questions:
        st.divider()
        st.markdown("Related")

        followup_container = st.container()
        with followup_container:
            for idx, followup_question in enumerate(
                st.session_state.followup_questions
            ):
                button_key = f"followup_{idx}_{hash(followup_question)}"

                if st.button(f"➕ {followup_question}", key=button_key):
                    st.session_state.messages.append(
                        {"role": "user", "content": followup_question}
                    )
                    success = handle_chat_response(
                        chat_client, user_id, followup_question, selected_domain
                    )

                    if success:
                        st.rerun()

    with st.sidebar:
        st.header("Retrieved Documents")
        if st.session_state.retrieved_documents:
            for idx, doc in enumerate(st.session_state.retrieved_documents):
                cleaned_doc = clean_document_text(doc)
                # Fix size limit later
                st.markdown(
                    f"**Document {idx + 1}:**\n\n{cleaned_doc[::]}\n",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("_No documents retrieved yet._")


def run():
    if "authenticated" in st.session_state and st.session_state.authenticated:
        main_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
