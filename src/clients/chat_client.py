from typing import Dict

import requests


class ChatClient:
    def __init__(self, base_url: str, api_version: str = "v1"):
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/chatbot"
        # self.client = httpx.AsyncClient()

    def chat(self, user_id: str, message: str, selected_domain: str) -> Dict:
        endpoint = f"{self.api_url}/chat"
        payload = {
            "user_id": user_id,
            "message": message,
            "selected_domain": selected_domain,
        }

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
