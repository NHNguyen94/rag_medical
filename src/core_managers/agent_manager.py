import os
from typing import Optional, List, Literal

import dotenv
from llama_index.agent.openai import OpenAIAgentWorker
from llama_index.core import PromptTemplate
from llama_index.core.agent import AgentRunner
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.indices.base import BaseIndex
from llama_index.core.response_synthesizers import ResponseMode
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
            system_prompt_template: Optional[PromptTemplate] = None,
            max_tokens: Optional[int] = None,
            reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
            temperature: Optional[float] = 0.7,
            similarity_top_k: Optional[int] = 5,
            force_use_tools: Optional[bool] = True,
    ):
        self.query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=ResponseMode.CONTEXT_ONLY,
            use_async=True,
            verbose=False,
            return_source=True,
        )
        self.tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.query_engine,
                name=ChatBotConfig.QUERY_ENGINE_TOOL,
                description=ChatBotConfig.QUERY_ENGINE_DESCRIPTION,
            )
        ]
        if force_use_tools:
            self.tool_choice = {"type": "function", "function": {"name": ChatBotConfig.QUERY_ENGINE_TOOL}}
        else:
            self.tool_choice = None
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            model=chat_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        self.worker = OpenAIAgentWorker.from_tools(
            tools=self.tools,
            llm=self.llm,
            system_prompt=system_prompt_template.template,
            verbose=False,
        )
        self.agent = AgentRunner(
            agent_worker=self.worker,
            llm=self.llm,
        )
        # self.agent = ReActAgent.from_tools(
        #     tools=self.tools,
        #     llm=self.llm,
        #     verbose=True,
        #     system_prompt=system_prompt_template.template if system_prompt_template else None,
        # )

    async def aget_stream_response(
            self,
            message: str,
            chat_history: Optional[List[ChatMessage]] = None,
    ) -> StreamingAgentChatResponse:
        return await self.agent.astream_chat(
            message=message,
            chat_history=chat_history,
            # Force the agent to use the query engine all the time
            tool_choice=self.tool_choice,
        )

    async def aget_nearest_documents(
            self,
            messages: str
    ) -> List[str]:
        response = await self.query_engine.aquery(messages)

        return [node.text for node in response.source_nodes]
