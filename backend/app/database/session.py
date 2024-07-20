# app/db/session.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from app.config import settings

engine = create_async_engine(settings.database_url, future=True, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncSession:  # type: ignore
    """
    The function `get_session` asynchronously yields a session object obtained from an async session
    context manager.
    """
    async with async_session() as session:
        yield session
