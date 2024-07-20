from sqlalchemy import Column, Integer, String
from app.db import Base

class Data(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True, index=True)
    moduleName = Column(String, index=True)
    imageUrl = Column(String)
