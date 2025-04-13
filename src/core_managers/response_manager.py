from llama_index.core.chat_engine.types import StreamingAgentChatResponse


class ResponseManager():
    def __init__(self):
        pass

    async def parse_stream_response(self, response: StreamingAgentChatResponse) -> str:
        response_str = ""
        async for token in response.async_response_gen():
            # Append token to UI later
            print(token, end="", flush=True)
            response_str += token
        print()
        return response_str
