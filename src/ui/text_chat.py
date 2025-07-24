import os

import streamlit as st
import torch

from src.clients.chat_client import ChatClient
from src.ui.utils import (
    login_or_signup,
    handle_chat_response,
    handle_chat_response_with_voice,
    define_customized_sys_prompt_path,
    define_customized_index_file_path,
)
from src.utils.enums import ChatBotConfig
from src.utils.helpers import clean_document_text, hash_string

torch.classes.__path__ = []


TRANSLATIONS = {
    "English": {
        "retrieved_documents": "Retrieved Documents",
        "download_csv": "Download Nearest Documents as CSV",
        "no_documents": "_No documents retrieved yet._",
        "disable_emotion": "Disable Emotion Recognition (Privacy)",
        "emotion_disabled": "🔒 Emotion Recognition: **DISABLED** (Privacy Mode)",
        "emotion_enabled": "🔓 Emotion Recognition: **ENABLED**",
        "logout": "Logout",
        "select_domain": "Select a medical domain",
        "select_model": "Select GPT Model",
        "selected_model_is": "Selected model is: {model}",
        "use_custom_prompt": "Use customized system prompt",
        "using_custom_prompt": "Using custom system prompt from:\n`{path}`",
        "use_custom_index": "Use customized index",
        "using_custom_index": "Using custom index from:\n`{path}`",
        "chat_input": "Type your message here...",
        "related": "Related",
        "generate_new_answer": "Generate New Answer",
        "like": "👍",
        "dislike": "👎",
        "thank_you_feedback": "Thank you for your feedback!",
        "sorry_not_helpful": "Sorry this answer did not help you.",
        "language": "Language",
        "welcome_message": "Hello, I'm your AI medical assistant. How can I help you today?",
        "select_chat_mode": "Select Chat Mode",
        "choose_input_type": "Choose input type:",
        "text_chat": "Text Chat 📝",
        "voice_chat": "Voice Chat 🎤",
        "doc_label": "Doc {num}:",
    },
    "French": {
        "retrieved_documents": "Documents récupérés",
        "download_csv": "Télécharger les documents les plus proches en CSV",
        "no_documents": "_Aucun document récupéré pour le moment._",
        "disable_emotion": "Désactiver la reconnaissance des émotions (Confidentialité)",
        "emotion_disabled": "🔒 Reconnaissance des émotions : **DÉSACTIVÉE** (Mode confidentialité)",
        "emotion_enabled": "🔓 Reconnaissance des émotions : **ACTIVÉE**",
        "logout": "Se déconnecter",
        "select_domain": "Sélectionnez un domaine médical",
        "select_model": "Sélectionnez le modèle GPT",
        "selected_model_is": "Le modèle sélectionné est : {model}",
        "use_custom_prompt": "Utiliser un prompt système personnalisé",
        "using_custom_prompt": "Prompt système personnalisé utilisé depuis :\n`{path}`",
        "use_custom_index": "Utiliser un index personnalisé",
        "using_custom_index": "Index personnalisé utilisé depuis :\n`{path}`",
        "chat_input": "Tapez votre message ici...",
        "related": "Connexe",
        "generate_new_answer": "Générer une nouvelle réponse",
        "like": "👍",
        "dislike": "👎",
        "thank_you_feedback": "Merci pour votre retour !",
        "sorry_not_helpful": "Désolé, cette réponse ne vous a pas aidé.",
        "language": "Langue",
        "welcome_message": "Bonjour, je suis votre assistant médical IA. Comment puis-je vous aider aujourd'hui ?",
        "select_chat_mode": "Sélectionnez le mode de chat",
        "choose_input_type": "Choisissez le type d'entrée :",
        "text_chat": "Chat texte 📝",
        "voice_chat": "Chat vocal 🎤",
        "doc_label": "Document {num} :",
    }
}


def main_app():
    user_id = st.session_state.get("hashed_username", "anonymous")
    st.title("📝 Text Chat - AI Medical Assistant")
    lang = st.session_state.get('selected_language', 'English')
    
    if st.button(TRANSLATIONS[lang]['logout']):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

    selected_domain = st.selectbox(TRANSLATIONS[lang]['select_domain'], ChatBotConfig.DOMAINS)

    available_models = [
        "gpt-3.5-turbo",
        "gpt-4",
        # Add more models as needed
    ]
    selected_model = st.selectbox(TRANSLATIONS[lang]['select_model'], available_models, index=0)
    st.write(TRANSLATIONS[lang]['selected_model_is'].format(model=selected_model))
    chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")

    use_custom_prompt = st.toggle(TRANSLATIONS[lang]['use_custom_prompt'], value=False)
    st.session_state['use_custom_prompt'] = use_custom_prompt
    if use_custom_prompt:
        customized_sys_prompt_path = define_customized_sys_prompt_path(user_id)
        st.info(TRANSLATIONS[lang]['using_custom_prompt'].format(path=customized_sys_prompt_path))
    else:
        customized_sys_prompt_path = None

    use_custom_index = st.toggle(TRANSLATIONS[lang]['use_custom_index'], value=False)
    if use_custom_index:
        customize_index_path = define_customized_index_file_path(user_id)
        st.info(TRANSLATIONS[lang]['using_custom_index'].format(path=customize_index_path))
    else:
        customize_index_path = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Remove all previous welcome messages (in any language)
    welcome_en = TRANSLATIONS['English']['welcome_message']
    welcome_fr = TRANSLATIONS['French']['welcome_message']
    st.session_state.messages = [
        m for m in st.session_state.messages
        if m['content'] not in [welcome_en, welcome_fr]
    ]
    # Add the welcome message in the selected language at the top
    st.session_state.messages.insert(0, {
        "role": "assistant",
        "content": TRANSLATIONS[lang]['welcome_message'],
    })

    print("MESSAGES:", st.session_state.messages)

    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []
    if "retrieved_documents" not in st.session_state:
        st.session_state.retrieved_documents = []

    print("RENDERING MESSAGE LOOP")
    for idx, message in enumerate(st.session_state.messages):
        print(f"IDX: {idx} | {message}")
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                feedback_key = f"feedback_{hash_string(message['content'])}_{idx}"
                feedback_given = st.session_state.get(feedback_key, None)
                if feedback_given is None:
                    col1, col2, col3 = st.columns([1, 1, 3])
                    like_clicked = col1.button(TRANSLATIONS[lang]['like'], key=f"like_{hash_string(message['content'])}_{idx}")
                    dislike_clicked = col2.button(TRANSLATIONS[lang]['dislike'], key=f"dislike_{hash_string(message['content'])}_{idx}")
                    if like_clicked or dislike_clicked:
                        feedback_type = "like" if like_clicked else "dislike"
                        st.session_state[feedback_key] = feedback_type
                        if like_clicked:
                            st.success(TRANSLATIONS[lang]['thank_you_feedback'])
                        else:
                            st.warning(TRANSLATIONS[lang]['sorry_not_helpful'])
                        st.rerun()  # <-- Add this line!
                elif feedback_given == "like":
                    st.success(TRANSLATIONS[lang]['thank_you_feedback'])
                elif feedback_given == "dislike":
                    st.warning(TRANSLATIONS[lang]['sorry_not_helpful'])
                    if st.button(TRANSLATIONS[lang]['generate_new_answer'], key=f"regen_{hash_string(message['content'])}_{idx}"):
                        with st.spinner("Generating a new answer..."):
                            # Find the previous user message before this assistant message
                            user_message = None
                            for prev in reversed(st.session_state.messages[:idx]):
                                if prev["role"] == "user":
                                    user_message = prev["content"]
                                    break
                            if user_message:
                                chat_client = ChatClient(base_url=os.getenv("API_URL"), api_version="v1")
                                # Get current settings
                                use_custom_prompt = st.session_state.get('use_custom_prompt', False)
                                customized_sys_prompt_path = None
                                if use_custom_prompt:
                                    customized_sys_prompt_path = define_customized_sys_prompt_path(user_id)
                                
                                import time
                                import random
                                cache_buster = f"regenerate_{int(time.time())}"
                                random_instructions = [
                                    "Please provide a comprehensive, detailed answer.",
                                    "Give a brief, concise response.",
                                    "Focus on practical advice and actionable steps.",
                                    "Explain this in simple terms for a general audience.",
                                    "Provide a clinical, medical perspective.",
                                    "Include both symptoms and prevention strategies.",
                                    "Focus on early warning signs and detection.",
                                    "Give a patient-friendly explanation with examples."
                                ]
                                random_instruction = random.choice(random_instructions)
                                full_prompt = f"{user_message}\n\n{random_instruction}"
                                print("[DEBUG] Generate New Answer:")
                                print("Prompt:", full_prompt)
                                print("Cache buster:", cache_buster)
                                print("Disable Emotion Recognition:", st.session_state.get('disable_emotion_recognition', False))
                                print("Selected Model:", selected_model)
                                print("Customized Sys Prompt Path:", customized_sys_prompt_path)
                                print("Language:", st.session_state.get('selected_language', 'English'))
                                handle_chat_response(
                                    chat_client,
                                    user_id,
                                    full_prompt,
                                    selected_domain,
                                    st.session_state.get('disable_emotion_recognition', False),
                                    selected_model,
                                    customized_sys_prompt_path,
                                    bypass_cache=True,
                                    language=st.session_state.get('selected_language', 'English'),
                                )
                                st.rerun()

    prompt = st.chat_input(TRANSLATIONS[lang]['chat_input'])

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Get current settings
        use_custom_prompt = st.session_state.get('use_custom_prompt', False)
        customized_sys_prompt_path = None
        if use_custom_prompt:
            customized_sys_prompt_path = define_customized_sys_prompt_path(user_id)
        
        handle_chat_response(
            chat_client,
            user_id,
            prompt,
            selected_domain,
            st.session_state.get('disable_emotion_recognition', False),
            selected_model,
            customized_sys_prompt_path,
            False,
            st.session_state.get('selected_language', 'English'),
        )
        st.rerun()  # <-- This forces the UI to update immediately

    if st.session_state.followup_questions:
        st.divider()
        st.markdown(TRANSLATIONS[lang]['related'])
        for idx, q in enumerate(st.session_state.followup_questions):
            if st.button(f"➕ {q}", key=f"followup_{idx}"):
                st.session_state.messages.append({"role": "user", "content": q})
                # Get current settings
                use_custom_prompt = st.session_state.get('use_custom_prompt', False)
                customized_sys_prompt_path = None
                if use_custom_prompt:
                    customized_sys_prompt_path = define_customized_sys_prompt_path(user_id)
                
                if handle_chat_response(
                    chat_client,
                    user_id,
                    q,
                    selected_domain,
                    st.session_state.get('disable_emotion_recognition', False),
                    selected_model,
                    customized_sys_prompt_path,
                    False,
                    st.session_state.get('selected_language', 'English'),
                ):
                    st.rerun()

    with st.sidebar:
        lang = st.session_state.get('selected_language', 'English')
        st.title(TRANSLATIONS[lang]['select_chat_mode'])
        st.markdown(TRANSLATIONS[lang]['choose_input_type'])
        chat_mode = st.radio(
            TRANSLATIONS[lang]['choose_input_type'],
            (TRANSLATIONS[lang]['text_chat'], TRANSLATIONS[lang]['voice_chat']),
            index=0
        )
        # Language selector with correct value and rerun on change
        new_lang = st.selectbox(
            TRANSLATIONS[lang]['language'],
            ["English", "French"],
            index=["English", "French"].index(lang),
            key="language_select"
        )
        if new_lang != lang:
            st.session_state['selected_language'] = new_lang
            st.rerun()
        st.header(TRANSLATIONS[lang]['retrieved_documents'])
        
        if st.session_state.retrieved_documents:
            for i, doc in enumerate(st.session_state.retrieved_documents):
                st.markdown(
                    f"**{TRANSLATIONS[lang]['doc_label'].format(num=i + 1)}** {clean_document_text(doc)}",
                    unsafe_allow_html=True,
                )
            # --- CSV Export Button ---
            import pandas as pd
            import io
            docs_df = pd.DataFrame({"Document": st.session_state.retrieved_documents})
            csv_buffer = io.StringIO()
            docs_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label=TRANSLATIONS[lang]['download_csv'],
                data=csv_buffer.getvalue(),
                file_name="nearest_documents.csv",
                mime="text/csv"
            )
        else:
            st.markdown(TRANSLATIONS[lang]['no_documents'])
        # Add toggle for emotion recognition
        disable_emotion = st.toggle(TRANSLATIONS[lang]['disable_emotion'], value=st.session_state.get('disable_emotion_recognition', False))
        st.session_state['disable_emotion_recognition'] = disable_emotion
        
        # Show current status
        if disable_emotion:
            st.info(TRANSLATIONS[lang]['emotion_disabled'])
        else:
            st.success(TRANSLATIONS[lang]['emotion_enabled'])


def run():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_or_signup()


if __name__ == "__main__":
    run()
