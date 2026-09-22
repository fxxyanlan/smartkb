"""SQLite 数据库（会话历史持久化）。

MVP 阶段用内存 dict 存会话；本模块预留扩展位。

TODO: 用 SQLAlchemy + Alembic 替换
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base
    Base = declarative_base()
    engine = create_engine("sqlite:///./data/askdocs.db")
    SessionLocal = sessionmaker(bind=engine)
"""