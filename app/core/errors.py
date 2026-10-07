"""领域异常：统一的业务错误类型。

分层约定：
    core / services 层只抛 AppError 及其子类（与 HTTP 无关）；
    api 层通过 main.py 注册的异常处理器，把 AppError 映射成对应的 HTTP 状态码。

这样业务代码不需要关心 HTTP，接口层也不需要写一堆 try/except。
"""
from __future__ import annotations


class AppError(Exception):
    """所有可预期业务错误的基类。"""

    status_code: int = 500
    default_message: str = "服务内部错误"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class UnsupportedFileTypeError(AppError):
    """文件类型不在白名单内。"""

    status_code = 400
    default_message = "不支持的文件类型"


class DocumentParseError(AppError):
    """文件损坏或解析失败（如把 exe 改名成 pdf）。"""

    status_code = 400
    default_message = "文档解析失败，文件可能已损坏或格式不正确"


class EmptyDocumentError(AppError):
    """文件为空，或未能提取到有效文本。"""

    status_code = 400
    default_message = "上传的文件为空，或未能从中提取到有效文本"


class FileTooLargeError(AppError):
    """文件超过大小上限。"""

    status_code = 413
    default_message = "文件超过大小限制"


class DocumentNotFoundError(AppError):
    """要操作的文档不存在。"""

    status_code = 404
    default_message = "文档不存在"


class LLMUnavailableError(AppError):
    """LLM 未配置或上游不可用。"""

    status_code = 503
    default_message = "模型服务暂时不可用"
