from langchain_community.chat_message_histories import RedisChatMessageHistory
from App.config import settings

class ShortTermMemoryManager:
    def __init__(self):
        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

    def get_chat_history(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=f"chat_history:{session_id}",
            url=self.redis_url
        )