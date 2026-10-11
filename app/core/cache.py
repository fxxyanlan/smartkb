"""检索结果缓存：自实现的 LRU + TTL，零新增依赖。

为什么自实现：项目约定检索层不引入额外第三方库，而这里的需求只是"同一问题短期内
重复提问不重复算"，`OrderedDict` 十几行即可覆盖，不需要 Redis/memcached。

两个容易踩的点：
    - 线程安全：检索通过 asyncio.to_thread 在线程池里跑，缓存会被并发读写，必须加锁。
    - 失效策略：缓存键里带上 store 的版本号（revision），文档增删后版本号变化，
      旧键自然不再命中，无需遍历清理。TTL 只用于兜底（例如底层模型被替换）。
"""
import threading
import time
from collections import OrderedDict

from app.config import get_settings


class RetrievalCache:
    """线程安全的 LRU + TTL 缓存。

    Args:
        maxsize: 最多缓存多少条；超出后淘汰最久未使用的键。
        ttl: 条目有效期（秒）；<=0 表示不设过期。
    """

    def __init__(self, maxsize: int = 256, ttl: float = 300.0) -> None:
        self._maxsize = max(1, maxsize)
        self._ttl = ttl
        self._data: OrderedDict[tuple, tuple[float, list]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple) -> list | None:
        """命中则返回结果列表的浅拷贝（防止调用方增删条目污染缓存），未命中返回 None。

        列表元素本身不深拷贝——检索结果按只读约定使用。
        """
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return None
            ts, value = item
            if self._ttl > 0 and now - ts > self._ttl:
                del self._data[key]
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return list(value)

    def put(self, key: tuple, value: list) -> None:
        """写入并维护 LRU 上限。"""
        with self._lock:
            self._data[key] = (time.monotonic(), list(value))
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """清空（测试与文档变更后主动失效用）。"""
        with self._lock:
            self._data.clear()

    @property
    def stats(self) -> dict:
        """命中统计，供健康检查 / 评估脚本观测缓存收益。"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


_cache: RetrievalCache | None = None


def get_retrieval_cache() -> RetrievalCache:
    """全局单例：所有检索器共用一份缓存。"""
    global _cache
    if _cache is None:
        s = get_settings()
        _cache = RetrievalCache(maxsize=s.cache_size, ttl=s.cache_ttl)
    return _cache
