from pydantic import BaseModel
from datetime import datetime


class CreateGuestChatRequest(BaseModel):
    name: str
    user_id: str


class CreateGuestChatResponse(BaseModel):
    id: str
    name: str
    user_id: str
    created_at: datetime
    updated_at: datetime


class SendGuestMessageRequest(BaseModel):
    content: str


class CreateUserChatRequest(BaseModel):
    name: str


class CreateUserChatResponse(BaseModel):
    id: str
    name: str
    user_id: str
    created_at: datetime
    updated_at: datetime


class SendUserMessageRequest(BaseModel):
    content: str


class UpdateUserChatRequest(BaseModel):
    name: str
