from fastapi import APIRouter
from sqlalchemy.orm import Session
from app.schemas import LoginPayload, Token, CreateUserPayload
from app.controllers.user_controller import create_user, login

router = APIRouter()


@router.post("/auth/signin", response_model=Token)
async def signin(payload: LoginPayload):
    return await login(payload)


@router.post("/auth/register", response_model=LoginPayload)
async def register_user(payload: CreateUserPayload):
    return await create_user(payload)
