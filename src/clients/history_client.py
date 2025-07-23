from typing import Dict, Optional

import requests


class HistoryClient:
    def __init__(self, base_url: str, api_version: str = "v1"):
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/history"

    def delete_chat_history(self, user_id: str) -> Dict:
        endpoint = f"{self.api_url}/delete_chat_history"
        payload = {"user_id": user_id}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def get_chat_history(self, user_id: str, limit: int = 10):
        response = requests.get(f"{self.api_url}/chat-history/{user_id}", params={"limit": limit})
        response.raise_for_status()
        return response.json()

    def delete_single_chat_message(self, chat_id: str):
        response = requests.delete(f"{self.api_url}/chat-history/message/{chat_id}")
        response.raise_for_status()
        return response.json()