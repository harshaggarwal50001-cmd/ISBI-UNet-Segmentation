from pydantic import BaseModel
from datetime import datetime

class PredictionOut(BaseModel):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
