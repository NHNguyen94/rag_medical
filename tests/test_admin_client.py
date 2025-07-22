from typing import Dict

from src.clients.admin_client import AdminClient
import pytest


class TestAdminClient:
    client = AdminClient(base_url="http://localhost:8000")

    def test_update_system_prompt(self):
        system_prompt = "Test system prompt"
        reasoning_effort = "high"
        temperature = 0.7
        similarity_top_k = 5
        yml_file = "tests/output/test_update_systemp_prompt.yml"

        response = self.client.update_system_prompt(
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            similarity_top_k=similarity_top_k,
            yml_file=yml_file,
        )

        assert isinstance(response, Dict)
        assert isinstance(response.get("system_prompt"), Dict)
