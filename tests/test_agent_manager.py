from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core_managers.agent_manager import AgentManager


@pytest.mark.asyncio
class TestAgentManager:
    agent = AgentManager(
        index=MagicMock(),
        chat_model="gpt-3.5-turbo",
    )

    async def test_aget_stream_response(self):
        self.agent.aget_stream_response = AsyncMock(return_value="Mocked Response")
        response = await self.agent.aget_stream_response("Hello")
        assert response == "Mocked Response"
        self.agent.aget_stream_response.assert_awaited_once_with("Hello")
