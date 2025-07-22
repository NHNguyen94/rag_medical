from typing import Optional, List

import redis

from src.utils.enums import RedisConfig


class RedisDBManager:
    def __init__(self):
        self.redis_client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )

    def set_value(self, key: str, value: str, ex: Optional[int] = None) -> None:
        if not key:
            raise ValueError("Key must not be empty")
        self.redis_client.set(name=key, value=value, ex=ex)

    def get_value(self, key: str) -> Optional[str]:
        if not key:
            raise ValueError("Key must not be empty")
        return self.redis_client.get(name=key)

    def delete_key(self, key: str) -> None:
        if not key:
            raise ValueError("Key must not be empty")
        self.redis_client.delete(key)

    def exists(self, key: str) -> bool:
        return self.redis_client.exists(key) == 1

    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        return self.redis_client.keys(pattern)

    def flush_db(self) -> None:
        self.redis_client.flushdb()

    def hset_value(
        self,
        name: str,
        key: str,
        value: str,
        ex: Optional[int] = RedisConfig.DEFAULT_EXPIRATION,
    ):
        if not name or not key:
            raise ValueError("Name and key must not be empty")
        self.redis_client.hset(name, key, value)
        if ex is not None:
            self.redis_client.expire(name, ex)

    def hget_value(self, name: str, key: str):
        if not name or not key:
            raise ValueError("Name and key must not be empty")
        return self.redis_client.hget(name, key)

    def hget_all(self, name: str):
        if not name:
            raise ValueError("Name must not be empty")
        return self.redis_client.hgetall(name)

    def hdel_field(self, name: str, key: str):
        if not name or not key:
            raise ValueError("Name and key must not be empty")
        self.redis_client.hdel(name, key)

    def hdel_hash(self, name: str):
        if not name:
            raise ValueError("Name must not be empty")
        self.redis_client.delete(name)

    def hincr_by(self, key: str, field: str, amount: int = 1):
        result = self.redis_client.hincrby(key, field, amount)
        return result

    def scan_iter_by_pattern(self, pattern: str) -> List[str]:
        return list(self.redis_client.scan_iter(pattern))

    def lpush(self, key: str, value: str):
        self.redis_client.lpush(key, value)

    def rpush(self, key: str, value: str):
        self.redis_client.rpush(key, value)

    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        return self.redis_client.lrange(key, start, end)

    def lrem(self, key: str, count: int, value: str):
        self.redis_client.lrem(key, count, value)

    def delete_list(self, key: str):
        self.redis_client.delete(key)

    def llen(self, key: str) -> int:
        return self.redis_client.llen(key)
