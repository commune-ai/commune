from datetime import timedelta
from fastapi import HTTPException, status
from app.models.user_model import User
from app.schemas import CreateUserPayload, LoginPayload
from app.auth import verify_password, create_access_token
from app.database.session import get_session


async def authenticate_user(email: str, password: str):
    async with get_session() as session:
        user = session.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user


def login(payload: LoginPayload):
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"token": access_token, "user": user}


async def create_user(payload: CreateUserPayload):
    async with get_session() as session:
        # Check if user already exists
        existing_user = session.query(User).filter(User.email == payload.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Create new user with default role 'general'
        new_user = User(
            name=payload.name,
            email=payload.email,
            hashed_password=payload.password,
            role="general",
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(
            new_user
        )  # Refresh to get the updated data including auto-generated fields

        return new_user
