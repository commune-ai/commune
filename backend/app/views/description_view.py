# app/views/description_view.py
from fastapi import APIRouter
from app.controllers.description_controller import (
    create_description,
    get_all_descriptions,
)
from pydantic import BaseModel
from typing import List

router = APIRouter()


class DescriptionRequest(BaseModel):
    name: str
    description: str


@router.post("/save-description/")
async def save_description(request: DescriptionRequest):
    return await create_description(request.name, request.description)


@router.get("/get-all-descriptions/")
async def get_all_descriptions_from_db():
    return await get_all_descriptions()
