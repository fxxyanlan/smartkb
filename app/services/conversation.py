"""会话存储：绑定的文档范围 + LRU 上限 + 历史轮数裁剪，供 ChatService 与 AgentService 共用。

除了历史本身，会话还记住两样东西，用于抑制「历史污染」：
    - doc_ids：本会话限定的文档范围，检索时按它过滤，旧文档物理上不可召回；
    - revision：进入会话时的向量库版本号，用于识别「会话中途上传/删除了文档」。

MVP 用内存 dict 存会话；生产应换成 SQLite / Redis。
"""
import logging
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """单个会话：问答历史 + 绑定的文档范围与数据版本。"""

    history: list[dict] = field(default_factory=list)
    #: 限定的文档范围；None 表示不限范围（全库检索）。
    doc_ids: list[str] | None = None
    #: 进入本会话时的向量库版本号；None 表示该向量库不提供版本号。
    revision: int | None = None


class ConversationStore:
    """内存会话表：按插入顺序，超量时淘汰最旧的，避免无限增长。"""

    def __init__(self, max_conversations: int = 200) -> None:
        self.conversations: OrderedDict[str, Conversation] = OrderedDict()
        self.max_conversations = max_conversations

    def get_or_create(
        self, conversation_id: str | None, max_conversations: int | None = None
    ) -> tuple[str, Conversation]:
        """取出或新建会话，并把最近使用的会话移到末尾（LRU 语义）。"""
        limit = max_conversations or self.max_conversations
        cid = conversation_id or self.new_id()
        if cid not in self.conversations:
            self.conversations[cid] = Conversation()
            if len(self.conversations) > limit:
                evicted, _ = self.conversations.popitem(last=False)
                logger.info("会话数超限，淘汰最旧会话 | cid=%s", evicted)
        self.conversations.move_to_end(cid)
        return cid, self.conversations[cid]

    def rebind(
        self,
        conversation: Conversation,
        doc_ids: Iterable[str] | None,
        revision: int | None,
    ) -> bool:
        """把会话对齐到当前文档范围与数据版本；不一致时清空历史。

        这是历史污染的第一道防线：会话中途上传/删除文档（版本号变化）或
        切换文档范围时，旧问答所依据的文档已不再可信，直接丢弃历史最安全。

        Returns:
            是否发生了历史重置。
        """
        scope = sorted(doc_ids) if doc_ids else None
        scope_changed = scope != conversation.doc_ids
        revision_changed = (
            revision is not None
            and conversation.revision is not None
            and revision != conversation.revision
        )
        reset = scope_changed or revision_changed
        if reset and conversation.history:
            logger.info(
                "文档范围或数据版本变化，已重置会话历史 | scope=%s revision=%s",
                scope,
                revision,
            )
            conversation.history.clear()
        conversation.doc_ids = scope
        conversation.revision = revision
        return reset

    def remember(
        self,
        history: list[dict],
        question: str,
        answer: str,
        max_turns: int | None = None,
    ) -> None:
        """把一轮问答追加进历史，并按轮数上限裁剪。"""
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        self.trim(history, max_turns or get_settings().max_history_turns)

    @staticmethod
    def new_id() -> str:
        return f"conv_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def trim(history: list[dict], max_turns: int) -> None:
        """保留最近 max_turns 轮对话。"""
        max_messages = max_turns * 2
        if len(history) > max_messages:
            del history[:-max_messages]
