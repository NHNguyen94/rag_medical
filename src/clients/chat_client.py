from typing import Dict, Optional

import requests


class ChatClient:
    def __init__(self, base_url: str = None, api_version: str = "v1"):
        if base_url is None:
            base_url = "http://localhost:8000"
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/chatbot"
        # self.client = httpx.AsyncClient()

    def chat(
        self,
        user_id: str,
        message: str,
        selected_domain: str,
        customized_sys_prompt_path: Optional[str] = None,
        customize_index_path: Optional[str] = None,
        model_name: Optional[str] = None,  # New parameter
        disable_emotion_recognition: Optional[bool] = False,
        bypass_cache: Optional[bool] = False,
        language: str = "English",
    ) -> Dict:
        endpoint = f"{self.api_url}/chat"
        payload = {
            "user_id": user_id,
            "message": message,
            "selected_domain": selected_domain,
            "customized_sys_prompt_path": customized_sys_prompt_path,
            "customize_index_path": customize_index_path,
            "language": language,
        }
        if model_name:
            payload["model_name"] = model_name
        if disable_emotion_recognition:
            payload["disable_emotion_recognition"] = disable_emotion_recognition
        if bypass_cache:
            import time
            payload["cache_buster"] = f"regenerate_{int(time.time())}"

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def voice_chat(self, user_id: str, audio_file: str, selected_domain: str) -> Dict:
        endpoint = f"{self.api_url}/voice_chat"
        payload = {
            "user_id": user_id,
            "audio_file": audio_file,
            "selected_domain": selected_domain,
        }

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def transcribe(self, audio_file: str) -> Dict:
        endpoint = f"{self.api_url}/transcribe"
        payload = {"audio_file": audio_file}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def text_to_speech(self, text: str, audio_path: str) -> Dict:
        endpoint = f"{self.api_url}/text_to_speech"
        payload = {"text": text, "audio_path": audio_path}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def submit_feedback(self, user_id: str, message: str, response: str, feedback_type: str) -> Dict:
        endpoint = f"{self.api_url}/feedback"
        payload = {
            "user_id": user_id,
            "message": message,
            "response": response,
            "feedback_type": feedback_type,
        }

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
