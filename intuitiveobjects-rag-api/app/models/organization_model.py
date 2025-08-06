from pydantic import BaseModel, Field
from datetime import datetime


class Organization(BaseModel):
    name: str
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())
