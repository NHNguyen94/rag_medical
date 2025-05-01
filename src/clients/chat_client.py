import httpx
import requests


class ChatClient:
    def __init__(self, base_url: str, api_version: str = "v1"):
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/chatbot"
        # self.client = httpx.AsyncClient()

    def chat(self, user_id: str, message: str, selected_domain: str) -> str:
        endpoint = f"{self.api_url}/chat"
        payload = {"user_id": user_id, "message": message, "selected_domain": selected_domain}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    #
    # async def achat(self, user_id: str, message: str, selected_domain: str) -> str:
    #     endpoint = f"{self.api_url}/chat"
    #     payload = {"user_id": user_id, "message": message, "domain": selected_domain}
    #
    #     response = await self.client.post(endpoint, json=payload)
    #
    #     if response.status_code == 200:
    #         return response.json()["response"]
    #     else:
    #         raise Exception(f"Error {response.status_code}: {response.text}")
