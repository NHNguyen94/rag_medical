import os
import asyncio
import dotenv
import streamlit as st

from src.clients.chat_client import ChatClient

dotenv.load_dotenv()

def run():
    st.title("AI-powered medical assistant")

    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
    user_id = "default_user_id"
    default_welcome_message = (
        "Hello, I'm your AI medical assistant. How can I help you today?"
    )

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
            # response = asyncio.run(chat_client.achat(user_id, prompt))
            response = chat_client.chat(user_id, prompt)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run()
