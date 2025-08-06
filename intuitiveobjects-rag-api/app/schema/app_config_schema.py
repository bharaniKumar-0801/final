from pydantic import BaseModel
from typing import List

class ModelOption(BaseModel):
    label: str
    value: str

class CreateAppConfigRequest(BaseModel):
    # llm_model: List[ModelOption]
    # embedding_model: List[ModelOption]

    llm_model: str
    embedding_model: str    
    temperature: float
    system_prompt: str = "You are a helpful AI assistant."  
    organization_id: str = "default_org"

    