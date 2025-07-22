import streamlit as st

from src.utils.directory_manager import DirectoryManager
from src.utils.enums import AudioConfig
from text_chat import run as run_text_chat
from voice_chat import run as run_voice_chat

directory_manager = DirectoryManager()


def main():
    directory_manager.create_dir_if_not_exists(AudioConfig.RECORDINGS_AUDIO_DIR)
    directory_manager.create_dir_if_not_exists(AudioConfig.OUTPUT_TTS_AUDIO_DIR)

    st.set_page_config(page_title="AI Medical Assistant", page_icon="🧠")

    st.sidebar.title("Select Chat Mode")
    chat_mode = st.sidebar.radio(
        "Choose input type:", ("Text Chat 📝", "Voice Chat 🎤")
    )

    if chat_mode.startswith("Text"):
        run_text_chat()
    elif chat_mode.startswith("Voice"):
        run_voice_chat()


if __name__ == "__main__":
    main()
