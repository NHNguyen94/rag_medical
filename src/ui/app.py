import os
# import sys

import dotenv
import streamlit as st

from src.clients.chat_client import ChatClient

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

dotenv.load_dotenv()


# TODO: Add and save chat history
def main(user_id: str, default_welcome_message: str, chat_client: ChatClient) -> None:
    st.title("AI-powered medical assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": default_welcome_message}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = chat_client.chat(user_id, prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
    user_id = "default_user_id"  # TODO: Work on this later for different users
    default_welcome_message = (
        "Hello, I'm your AI medical assistant. How can I help you today?"
    )
    main(user_id, default_welcome_message, chat_client)
