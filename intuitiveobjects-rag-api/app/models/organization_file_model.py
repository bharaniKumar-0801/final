from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId
from typing import List
class OrganizationFile(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    organization_id: str
    category_id: str
    file_name: str
    file_type: str = Field(default="")
    file_size: int = Field(default=0)
    type: str = Field(default="local")
    drive_file_id: str = Field(default="")
    drive_web_link: str = Field(default="")
    tags: List[str] = Field(default=[])
    storage_id: str = Field(default="")
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())