# app/models/description_model.py
from sqlalchemy import Column, Integer, String, Text
from app.db import Base


class Description(Base):
    __tablename__ = "descriptions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)
