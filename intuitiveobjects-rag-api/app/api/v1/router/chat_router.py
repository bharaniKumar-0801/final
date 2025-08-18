from fastapi import APIRouter, Depends
from app.schema.chat_schema import (
    CreateGuestChatRequest,
    SendGuestMessageRequest,
    CreateUserChatRequest,
    SendUserMessageRequest,
    UpdateUserChatRequest,
)
import app.services.chat_service as chat_service
from app.utils.auth import get_current_user
from app.backend import ask_question

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.pipeline.rag_utils import process_query



chat_router = APIRouter()

# User Chat Endpoints
@chat_router.post("/user")
async def create_user_chat(
    chat: CreateUserChatRequest, user_id: str = Depends(get_current_user)
):
    """Create a new user chat."""
    result = await chat_service.create_user_chat(chat, user_id)
    return {"success": True, "message": "Chat created successfully", "data": result}


@chat_router.get("/user")
async def get_user_chats(user_id: str = Depends(get_current_user)):
    """Retrieve all user chats."""
    result = await chat_service.get_user_chats(user_id)
    return {"success": True, "message": "Chats retrieved successfully", "data": result}


@chat_router.get("/{chat_id}/user")
async def get_user_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    """Retrieve a specific user chat by chat ID."""
    result = await chat_service.get_user_chat(chat_id, user_id)
    return {"success": True, "message": "Chat retrieved successfully", "data": result}


@chat_router.get("/{chat_id}/user/messages")
async def get_user_chat_messages(
    chat_id: str, user_id: str = Depends(get_current_user)
):
    """Retrieve messages from a specific user chat."""
    result = await chat_service.get_user_chat_messages(chat_id, user_id)
    return {
        "success": True,
        "message": "Messages retrieved successfully",
        "data": result,
    }


@chat_router.delete("/{chat_id}/user")
async def delete_user_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    """Delete a specific user chat by chat ID."""
    result = await chat_service.delete_user_chat(chat_id, user_id)
    return {"success": True, "message": "Chat deleted successfully", "data": result}


@chat_router.put("/{chat_id}/user")
async def update_user_chat(chat_id: str, chat: UpdateUserChatRequest, user_id: str = Depends(get_current_user)):
    """Update a specific user chat by chat ID."""
    result = await chat_service.update_user_chat(chat_id, chat, user_id)
    return {"success": True, "message": "Chat updated successfully", "data": result}



@chat_router.post("/{chat_id}/user/message")
async def send_user_message(
    chat_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    try:
        data = await request.json()
        question = data.get("content")
        if not question:
            raise ValueError("Content is required")

        # Save user message to chat history
        user_msg = await chat_service.send_user_message(
            chat_id, 
            user_id, 
            SendUserMessageRequest(content=question)
        )

        return {
            "success": True,
            "message": "Message sent successfully",
            "data": user_msg,
        }

    except Exception as e:
        print(f"Error in send_user_message: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )



# Guest Chat Endpoints
@chat_router.post("/guest")
async def create_guest_chat(chat: CreateGuestChatRequest):
    """Create a new guest chat."""
    result = await chat_service.create_guest_chat(chat)
    return {
        "success": True,
        "message": "Guest chat created successfully",
        "data": result,
    }


@chat_router.get("/guest/{user_id}")
async def get_guest_chats(user_id: str):
    """Retrieve all guest chats for a specific user."""
    result = await chat_service.get_guest_chats(user_id)
    return {
        "success": True,
        "message": "Guest chats retrieved successfully",
        "data": result,
    }


@chat_router.get("/{chat_id}/guest/{user_id}")
async def get_guest_chat(chat_id: str, user_id: str):
    """Retrieve a specific guest chat by chat ID and user ID."""
    result = await chat_service.get_guest_chat(chat_id, user_id)
    return {
        "success": True,
        "message": "Guest chat retrieved successfully",
        "data": result,
    }


@chat_router.get("/{chat_id}/guest/{user_id}/messages")
async def get_guest_chat_messages(chat_id, user_id):
    """Retrieve messages from a specific guest chat."""
    result = await chat_service.get_guest_chat_messages(chat_id, user_id)
    return {
        "success": True,
        "message": "Messages retrieved successfully",
        "data": result,
    }


@chat_router.delete("/{chat_id}/guest/{user_id}")
async def delete_guest_chat(chat_id: str, user_id: str):
    """Delete a specific guest chat by chat ID and user ID."""
    result = await chat_service.delete_guest_chat(chat_id, user_id)
    return {"success": True, "message": "Chat deleted successfully", "data": result}


@chat_router.post("/{chat_id}/guest/{user_id}/message")
async def send_guest_message(
    chat_id: str, user_id: str, message: SendGuestMessageRequest
):
    """Send a message in a specific guest chat."""
    result = await chat_service.send_guest_message(chat_id, user_id, message)
    return {"success": True, "message": "Message sent successfully", "data": result}
