from pydantic import BaseModel, Field
from datetime import datetime


class Token(BaseModel):
    user_id: str
    token: str
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())
