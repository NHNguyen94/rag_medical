import os
from typing import Optional, List, Literal

import dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.indices.base import BaseIndex
from llama_index.core.memory import BaseMemory
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.llms.openai import OpenAI

from src.utils.enums import ChatBotConfig

dotenv.load_dotenv()


class AgentManager:
    def __init__(
        self,
        # For vector database
        index: BaseIndex,
        chat_model: str = ChatBotConfig.DEFAULT_CHAT_MODEL,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        temperature: Optional[float] = 0.7,
        memory: Optional[BaseMemory] = None
    ):
        self.query_engine = index.as_query_engine()
        self.tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.query_engine,
                name="Query Engine",
                description="Query engine for the index",
            )
        ]
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                model=chat_model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
            ),
            memory=memory,
        )

    async def aget_stream_response(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> StreamingAgentChatResponse:
        return await self.agent.astream_chat(
            message=message,
            chat_history=chat_history
        )
