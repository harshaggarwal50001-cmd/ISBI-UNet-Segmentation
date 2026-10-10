from sqlalchemy import Column, Integer, LargeBinary, DateTime
from datetime import datetime
from backend.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    original_image = Column(LargeBinary, nullable=False)
    predicted_image = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
