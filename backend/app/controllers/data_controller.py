from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.data_model import Data
from typing import List
from sqlalchemy.future import select
from app.database.session import get_session


async def save_data_to_db(moduleName: str, imageUrl: str):
    async with get_session() as session:
        # Check if moduleName already exists
        result = await session.execute(select(Data).filter(Data.moduleName == moduleName))
        existing_data = result.scalars().first()

        if existing_data:
            raise HTTPException(status_code=400, detail="moduleName already exists")

        # Save new data if moduleName does not exist
        new_data = Data(moduleName=moduleName, imageUrl=imageUrl)
        session.add(new_data)
        try:
            await session.commit()
            return {"message": "Data saved successfully"}
        except Exception as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=str(e))

async def get_all_data() -> List[Data]:
    async with get_session() as session:
        result = await session.execute(select(Data))
        return result.scalars().all()
