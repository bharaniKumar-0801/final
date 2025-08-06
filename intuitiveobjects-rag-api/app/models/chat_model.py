from pydantic import BaseModel, Field
from datetime import datetime


class Chat(BaseModel):
    user_id: str
    name: str
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())
