import os
import wave
import streamlit as st
import torch
from st_audiorec import st_audiorec

from src.clients.chat_client import ChatClient
from src.ui.utils import (
    login_or_signup,
    handle_chat_response_with_voice,
    define_customized_sys_prompt_path,
    define_customized_index_file_path,
)
from src.utils.enums import ChatBotConfig, AudioConfig, AdminConfig
from src.utils.date_time_manager import DateTimeManager
from src.utils.helpers import clean_document_text

datetime_manager = DateTimeManager()
torch.classes.__path__ = []


def main_app():
    st.title("🎤 Voice Chat - AI Medical Assistant")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

    selected_domain = st.selectbox("Select a medical domain", ChatBotConfig.DOMAINS)
    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
    user_id = st.session_state.get("hashed_username", "default_user_id")

    use_custom_prompt = st.toggle("Use customized system prompt", value=False)
    if use_custom_prompt:
        customized_sys_prompt_path = define_customized_sys_prompt_path(user_id)
        customized_sys_prompt_path = os.path.join(
            AdminConfig.CUSTOMIZED_SYSTEM_PROMPT_DIR, customized_sys_prompt_path
        )
        st.info(f"Using custom system prompt from:\n`{customized_sys_prompt_path}`")
    else:
        customized_sys_prompt_path = None

    use_custom_index = st.toggle("Use customized index", value=False)
    if use_custom_index:
        customize_index_path = define_customized_index_file_path(user_id)
        st.info(f"Using custom index from:\n`{customize_index_path}`")
    else:
        customize_index_path = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello, I'm your AI medical assistant. How can I help you today?",
            }
        ]
    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []
    if "retrieved_documents" not in st.session_state:
        st.session_state.retrieved_documents = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("Click below to record your message:")

    audio_bytes = st_audiorec()

    if audio_bytes:
        st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_ready = True

    prompt = None

    if audio_bytes:
        recordings_audio_dir = AudioConfig.RECORDINGS_AUDIO_DIR
        timestamp = datetime_manager.get_current_local_time_str()
        wav_filename = f"{user_id}_recording_{timestamp}.wav"
        wav_path = os.path.join(recordings_audio_dir, wav_filename)

        # save the audio bytes directly to a file
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # st.success(f"Audio saved: {wav_path}")
        # put this here to not block the session state
        # st.audio(wav_path, format=AudioConfig.AUDIO_FORMAT)

        prompt = chat_client.transcribe(wav_path).get("transcription")
        print(f"Transcribed prompt: {prompt}")

        st.session_state.audio_ready = False

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        handle_chat_response_with_voice(
            chat_client,
            user_id,
            prompt,
            selected_domain,
            customized_sys_prompt_path,
            customize_index_path,
        )

    if st.session_state.followup_questions:
        st.divider()
        st.markdown("Related")
        for idx, q in enumerate(st.session_state.followup_questions):
            if st.button(f"➕ {q}", key=f"followup_{idx}"):
                st.session_state.messages.append({"role": "user", "content": q})
                if handle_chat_response_with_voice(
                    chat_client,
                    user_id,
                    q,
                    selected_domain,
                    customized_sys_prompt_path,
                    customize_index_path,
                ):
                    st.rerun()

    with st.sidebar:
        st.header("Retrieved Documents")
        if st.session_state.retrieved_documents:
            for i, doc in enumerate(st.session_state.retrieved_documents):
                st.markdown(
                    f"**Doc {i + 1}:** {clean_document_text(doc)}",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("_No documents retrieved yet._")


def run():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
