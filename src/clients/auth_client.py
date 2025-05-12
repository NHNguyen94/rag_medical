import httpx
import requests

class AuthClient:
    def __init__(self, base_url: str, api_version: str = "v1"):
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/auth"
        # self.client = httpx.AsyncClient()

    def login(self, username: str, password: str) -> str:
        endpoint = f"{self.api_url}/login"
        payload = {"username": username, "password": password}

        response = requests.post(endpoint, json=payload)
        print(f"Response: {response.status_code} {response.text}")

        if response.status_code == 200:
            return response.json()["message"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def register(self, username: str, password: str) -> str:
        endpoint = f"{self.api_url}/register"
        payload = {"username": username, "password": password}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()["message"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")