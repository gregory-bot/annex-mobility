"""Async SQLAlchemy engine with connection pooling tuned for scale."""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Connection pool tuned for production:
# - pool_size=20, max_overflow=40 handles ~1000 concurrent users per instance
# - Use multiple instances behind a load balancer for 1M users
# - Switch DATABASE_URL to PostgreSQL (asyncpg) for production
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=(settings.APP_ENV == "development"),
    pool_pre_ping=True,
    # SQLite: no pool needed | PostgreSQL: tune these
    **(
        {}
        if settings.DATABASE_URL.startswith("sqlite")
        else {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_timeout": 30,
            "pool_recycle": 1800,
        }
    ),
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
