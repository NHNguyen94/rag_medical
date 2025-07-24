import streamlit as st

from src.utils.directory_manager import DirectoryManager
from src.utils.enums import AudioConfig, AdminConfig
from text_chat import run as run_text_chat
from voice_chat import run as run_voice_chat

directory_manager = DirectoryManager()


def main():
    directory_manager.create_dir_if_not_exists(AudioConfig.RECORDINGS_AUDIO_DIR)
    directory_manager.create_dir_if_not_exists(AudioConfig.OUTPUT_TTS_AUDIO_DIR)
    directory_manager.create_dir_if_not_exists(AdminConfig.CUSTOMIZED_SYSTEM_PROMPT_DIR)
    directory_manager.create_dir_if_not_exists(AdminConfig.CUSTOMIZED_CSV_DIR)
    directory_manager.create_dir_if_not_exists(AdminConfig.CUSTOMIZED_INDEX_DIR)

    st.set_page_config(page_title="AI Medical Assistant", page_icon="🧠")

    # Removed sidebar rendering here to avoid duplication
    # st.sidebar.title("Select Chat Mode")
    # chat_mode = st.sidebar.radio(
    #     "Choose input type:", ("Text Chat 📝", "Voice Chat 🎤")
    # )

    # Use a simple session state or default to text chat
    if 'chat_mode' not in st.session_state:
        st.session_state['chat_mode'] = 'Text Chat 📝'

    # You can set chat_mode from the sidebar in text_chat.py/voice_chat.py
    if st.session_state['chat_mode'].startswith("Text"):
        run_text_chat()
    elif st.session_state['chat_mode'].startswith("Voice"):
        run_voice_chat()


if __name__ == "__main__":
    main()
