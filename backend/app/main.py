# app/main.py
from fastapi import FastAPI
from app.views.description_view import router as description_router
from app.views.data_view import router as data_router
from app.views.user_view import router as user_router

from app.db import engine, Base
from fastapi.middleware.cors import CORSMiddleware  # If you need CORS support
import uvicorn

# Create FastAPI app instance
app = FastAPI()

# CORS middleware to allow all origins (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Event handler to create database tables on startup
@app.on_event("startup")
async def startup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Include routers
app.include_router(description_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(user_router, prefix="/api")

# If running this script directly, start Uvicorn server
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
