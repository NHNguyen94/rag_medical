import asyncio

from src.services.chat_bot_service import ChatBotService


async def main():
    user_id = "user_123"
    query = "Write a 100 words summary of the book 'Harry Potter and the Philosopher's Stone'."
    print(f"User ID: {user_id}\n")
    print(f"Query: {query}\n")
    print("Response:\n")
    chat_bot_service = ChatBotService(user_id=user_id)
    response = await chat_bot_service.achat(query)

if __name__ == "__main__":
    asyncio.run(main())
