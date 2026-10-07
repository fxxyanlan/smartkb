"""应用配置：统一管理所有环境变量与路径。

所有可配置项都在 Settings 类里，通过 .env 文件覆盖。
任何模块获取配置都用 get_settings()，保证全局只有一份。
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """所有配置项。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- LLM -----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    # ----- Embedding -----
    embedding_model: str = "./data/models/models/BAAI--bge-small-zh-v1.5/snapshots/master"
    embedding_device: str = "cpu"

    # ----- Vector Store -----
    vector_store_type: str = "chroma"
    chroma_persist_dir: str = "./data/vector_store"

    # ----- Application -----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    # 允许跨域的来源。默认 * 仅用于开发；生产应显式配置具体域名。
    cors_origins: list[str] = ["*"]

    # ----- 文件上传 -----
    upload_dir: str = "./data/documents"
    max_upload_size_mb: int = 20

    # ----- RAG -----
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    max_history_turns: int = 6

    @property
    def llm_configured(self) -> bool:
        """是否已配置 LLM 密钥。"""
        return bool(self.deepseek_api_key.strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        """上传目录（自动转绝对路径）。"""
        p = Path(self.upload_dir)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def project_root(self) -> Path:
        """项目根目录。"""
        return Path(__file__).resolve().parent.parent

    @property
    def chroma_path(self) -> Path:
        """Chroma 持久化目录（自动转绝对路径）。"""
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = self.project_root / p
        return p


@lru_cache
def get_settings() -> Settings:
    """单例：全局共享一份配置。"""
    return Settings()
