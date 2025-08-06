import uuid
from pydantic import BaseModel, Field
from datetime import datetime


class User(BaseModel):
    email: str
    password: str
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())
