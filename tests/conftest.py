"""pytest 共享 fixture。"""
import pytest

from app.config import get_settings


@pytest.fixture
def settings():
    """全局配置。"""
    return get_settings()


@pytest.fixture
def sample_question() -> str:
    return "这份文档讲了什么?"