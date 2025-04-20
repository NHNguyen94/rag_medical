import pytest
from src.core_managers.agent_manager import AgentManager
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def agent_manager():
    return AgentManager(index=MagicMock(), chat_model="gpt-3.5-turbo")


@pytest.mark.asyncio
class TestAgentManager:
    async def test_aget_stream_response(self, agent_manager):
        agent_manager.aget_stream_response = AsyncMock(return_value="Mocked Response")
        response = await agent_manager.aget_stream_response("Hello")
        assert response == "Mocked Response"
        agent_manager.aget_stream_response.assert_awaited_once_with("Hello")
