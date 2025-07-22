from typing import Dict

import requests


class AdminClient:
    def __init__(self, base_url: str, api_version: str = "v1"):
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/admin"

    def update_system_prompt(
        self,
        system_prompt: str,
        reasoning_effort: str,
        temperature: float,
        similarity_top_k: int,
        yml_file: str,
    ) -> Dict:
        endpoint = f"{self.api_url}/update_system_prompt"
        payload = {
            "system_prompt": system_prompt,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "similarity_top_k": similarity_top_k,
            "yml_file": yml_file,
        }

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
