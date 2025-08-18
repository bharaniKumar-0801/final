from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
from app.schema.app_config_schema import ModelOption

class AppConfig(BaseModel):
    llm_models: List[ModelOption]
    embedding_models: List[ModelOption]
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())