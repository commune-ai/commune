# app/schemas.py
from pydantic import BaseModel


class LoginPayload(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str

    class Config:
        orm_mode = True


class Token(BaseModel):
    token: str
    user: UserOut


class CreateUserPayload(BaseModel):
    name: str
    email: str
    password: str
    # Add other fields as necessary
