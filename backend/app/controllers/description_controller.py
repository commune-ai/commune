# app/controllers/description_controller.py
from fastapi import HTTPException
from app.models.description_model import Description
from sqlalchemy.future import select
from typing import List
from app.database.session import get_session


async def create_description(name: str, description: str):
    async with get_session() as session:
        new_description = Description(name=name, text=description)
        session.add(new_description)
        try:
            await session.commit()
            return {"message": "Description saved successfully"}
        except Exception as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=str(e))


async def get_all_descriptions() -> List[Description]:
    async with get_session() as session:
        result = await session.execute(select(Description))
        return result.scalars().all()
