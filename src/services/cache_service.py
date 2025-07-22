import hashlib
import json
from typing import Optional, Dict

from src.core_managers.redis_db_manager import RedisDBManager


class CacheService:
    def __init__(self):
        self.redis_manager = RedisDBManager()

    def _make_cache_key(self, request: Dict) -> str:
        request_str = json.dumps(request, sort_keys=True)
        key_hash = hashlib.sha256(request_str.encode()).hexdigest()
        return f"chat_cache:{key_hash}"

    def cache_request_and_response(self, request: Dict, response: Dict) -> None:
        cache_key = self._make_cache_key(request)
        response_str = json.dumps(response)

        self.redis_manager.set_value(key=cache_key, value=response_str)

    def get_cached_response(self, request: Dict) -> Optional[Dict]:
        cache_key = self._make_cache_key(request)
        cached_value = self.redis_manager.get_value(cache_key)

        if cached_value:
            return json.loads(cached_value)
        return None
