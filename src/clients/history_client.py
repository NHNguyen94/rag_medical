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