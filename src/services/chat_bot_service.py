from src.core_managers import AgentManager, VectorStoreManager, ResponseManager


class ChatBotService:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_store_manager = VectorStoreManager(user_id)
        # Handle the vector store initialization later
        self.index = self.vector_store_manager.build_index([])
        self.agent = AgentManager(self.index)
        self.response_manager = ResponseManager()

    async def achat(self, message: str) -> str:
        response = await self.agent.aget_stream_response(message)
        response_str = await self.response_manager.parse_stream_response(response)
        return response_str
