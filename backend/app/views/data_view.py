from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.controllers.description_controller import get_all_data
from app.db import SessionLocal
from app.controllers.data_controller import save_data_to_db
from typing import AsyncGenerator, List
from pydantic import BaseModel
from contextlib import asynccontextmanager

router = APIRouter()


class DataRequest(BaseModel):
    moduleName: str
    imageUrl: str


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            await session.close()


@router.post("/save-data/")
async def save_data(request: DataRequest, session: AsyncSession = Depends(get_session)):
    return await save_data_to_db(session, request.moduleName, request.imageUrl)


@router.get("/get-all-data/")
async def get_all_data_from_db(session: AsyncSession = Depends(get_session)):
    return await get_all_data(session)
