"""统一日志配置。"""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志格式。

    日志格式：时间 | 级别 | logger 名称 | 消息
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )