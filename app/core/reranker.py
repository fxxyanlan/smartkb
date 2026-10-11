"""重排序（Rerank）：粗召回之后再对候选精排，提升首位准确率。

为什么需要：向量检索是"双塔"结构（问题、片段各自编码，只在最后算余弦），
BM25 只看词频统计——两者都只能保证"相关片段落在候选池里"，但不保证排在最前。
交叉编码器（CrossEncoder）把 (问题, 片段) 一起送进模型、直接建模二者的交互，
排序质量显著更高；代价是无法预先建索引，只能对小候选池逐个打分。

因此流程是两段式：粗召回 N 个候选 → 精排取 top_k。N 由 rerank_candidates 控制。

设计要点：
    - Reranker 是抽象接口：rerank(query, candidates, top_k) -> list[dict]
    - CrossEncoderReranker 懒加载模型，且**加载/打分失败自动降级**（退回原顺序），
      保证「模型没准备好」不会让整个问答链路不可用；
    - NoopReranker 是"不重排"的显式兜底（未启用 / 模型不可用时）；
    - 精排分数写入片段的 rerank_score 字段，便于前端与评估观察。
"""
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """重排接口。实现方只需保证：输入候选列表，输出重排后的前 top_k 个。"""

    #: 模型是否真的可用（Noop / 降级后为 False）。评估脚本据此决定是否计入对照。
    available: bool = False

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        raise NotImplementedError


class NoopReranker(Reranker):
    """不重排：直接截断候选池。用于未启用重排或模型不可用时的兜底。"""

    available = False

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return candidates[:top_k]


class CrossEncoderReranker(Reranker):
    """交叉编码器重排：懒加载 + 失败降级。

    模型加载与打分都在 try/except 内，任何异常都只记录日志并退回"不重排"，
    绝不向上抛——检索链路的可用性优先于排序质量。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model=None,
        batch_size: int = 16,
    ) -> None:
        s = get_settings()
        self._model_name = model_name or s.rerank_model
        self._device = device or s.rerank_device
        self._batch_size = batch_size
        self._model = model  # 允许注入，便于测试
        self._load_failed = False

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        return self._ensure_model() is not None

    def _ensure_model(self):
        """惰性加载模型；失败则永久降级（不重复尝试）。"""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
            logger.info("重排序模型已加载 | model=%s", self._model_name)
        except Exception:  # noqa: BLE001 - 加载失败一律降级，不能影响可用性
            self._load_failed = True
            logger.warning(
                "重排序模型加载失败，已降级为不重排 | model=%s",
                self._model_name,
                exc_info=True,
            )
        return self._model

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        model = self._ensure_model()
        if model is None:
            return candidates[:top_k]

        pairs = [(query, c.get("chunk_text", "")) for c in candidates]
        try:
            scores = model.predict(pairs, batch_size=self._batch_size)
        except Exception:  # noqa: BLE001 - 打分失败退回原顺序
            logger.warning("重排序打分失败，已退回原顺序", exc_info=True)
            return candidates[:top_k]

        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda r: r["rerank_score"], reverse=True)
        return scored[:top_k]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """全局单例：未启用重排时返回 NoopReranker（不加载任何模型）。"""
    global _reranker
    if _reranker is None:
        if get_settings().rerank_enabled:
            _reranker = CrossEncoderReranker()
        else:
            _reranker = NoopReranker()
    return _reranker
