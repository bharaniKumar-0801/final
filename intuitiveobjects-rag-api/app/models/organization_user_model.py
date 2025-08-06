from pydantic import BaseModel, Field
from datetime import datetime


class OrganizationUser(BaseModel):
    email: str
    password: str
    category_id: str
    organization_id: str
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())
