from app.schema.chat_schema import (
    CreateGuestChatRequest,
    SendGuestMessageRequest,
    CreateUserChatRequest,
    SendUserMessageRequest,
    UpdateUserChatRequest,
)
from app.models.chat_model import Chat
from app.db.mongodb import chat_collection
from fastapi import HTTPException
from app.serializers.chat_serializers import chatEntity, chatListEntity
from app.models.message_model import Message, MessageRole
from app.db.mongodb import message_collection
from app.db.mongodb import user_query_collection
from app.serializers.message_serializers import messageListEntity, messageEntity
from bson.objectid import ObjectId
from app.utils.response_generator import generate_ai_response
from app.backend import ask_question  
from app.pipeline.rag_pipeline import expand_query_with_context
# from app.llm import expand_user_query  # Adjust import path if needed
# from app.db.mongodb import message_collection  # Adjust import path if needed
import logging
logger = logging.getLogger(__name__)


"""
    Create Guest Chat
"""


async def create_guest_chat(chat: CreateGuestChatRequest):
    try:
        new_chat = Chat(
            user_id=chat.user_id,
            name=chat.name,
        )

        result = await chat_collection().insert_one(new_chat.model_dump())

        if result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to create chat")

        created_chat = await chat_collection().find_one({"_id": result.inserted_id})

        return chatEntity(created_chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Get Guest Chats
"""


async def get_guest_chats(user_id: str):
    try:
        chats = await chat_collection().find({"user_id": user_id}).to_list(length=100)
        return chatListEntity(chats)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


"""
    Get Guest Chat
"""


async def get_guest_chat(chat_id: str, user_id: str):
    try:
        chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        return chatEntity(chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Get Guest Chat Messages
"""


async def get_guest_chat_messages(chat_id: str, user_id: str):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        messages = (
            await message_collection().find({"chat_id": chat_id}).to_list(length=100)
        )
        return messageListEntity(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Delete Guest Chat
"""


async def delete_guest_chat(chat_id: str, user_id: str):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        result = await chat_collection().delete_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"message": "Chat deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Send Guest Message
"""


async def send_guest_message(
    chat_id: str, user_id: str, message: SendGuestMessageRequest
):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        

        request_message = Message(
            chat_id=chat_id, content=message.content, role=MessageRole.user
        )

        ai_response = generate_ai_response(message.content)

        response_message = Message(
            chat_id=chat_id, content=ai_response, role=MessageRole.assistant
        )

        request_message_result = await message_collection().insert_one(
            request_message.model_dump()
        )

        if request_message_result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to send message")

        response_message_result = await message_collection().insert_one(
            response_message.model_dump()
        )

        if response_message_result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to send message")

        response_message = await message_collection().find_one(
            {"_id": response_message_result.inserted_id}
        )
        request_message = await message_collection().find_one(
            {"_id": request_message_result.inserted_id}
        )

        return {
            "response_message": messageEntity(response_message),
            "request_message": messageEntity(request_message),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Create User Chat
"""


async def create_user_chat(chat: CreateUserChatRequest, user_id: str):
    try:
        new_chat = Chat(user_id=user_id, name=chat.name)

        chat_count = await chat_collection().count_documents({"user_id": user_id})

        new_chat.name = f"Chat {chat_count + 1}"

        result = await chat_collection().insert_one(new_chat.model_dump())

        if result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to create chat")

        created_chat = await chat_collection().find_one({"_id": result.inserted_id})

        return chatEntity(created_chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Get User Chats
"""


async def get_user_chats(user_id: str):
    try:
        chats = await chat_collection().find({"user_id": user_id}).to_list(length=100)
        return chatListEntity(chats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Get User Chat
"""


async def get_user_chat(chat_id: str, user_id: str):
    try:
        chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        return chatEntity(chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Get User Chat Messages
"""


async def get_user_chat_messages(chat_id: str, user_id: str):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        messages = (
            await message_collection().find({"chat_id": chat_id}).to_list(length=100)
        )
        return messageListEntity(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Delete User Chat
"""


async def delete_user_chat(chat_id: str, user_id: str):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        result = await chat_collection().delete_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chatEntity(existing_chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Send User Message
"""


async def send_user_message(
    chat_id: str, user_id: str, message: SendUserMessageRequest
):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        expanded_query = await expand_query_with_context(chat_id, message.content) 
# user query 
        user_message =  request_message = Message(
            chat_id=chat_id, content=message.content, role=MessageRole.user
        ) 
        
        logger.info(f"[send_user_message] user_message: {user_message}")


# expand the user query with context from the chat history
        request_message = Message(
            chat_id=chat_id, content=expanded_query, role=MessageRole.user
        )

        # Expand the user query with context from the chat history
        # expanded_query = await expand_query_with_context(chat_id, message.content)

        # ai_response = generate_ai_response(message.content)
        # ai_response = await ask_question(expanded_query)

        ai_response = await ask_question(expanded_query, chat_id)

        if isinstance(ai_response, dict):
            ai_response = ai_response.get("response", "").strip()

        response_message = Message(
            chat_id=chat_id, content=ai_response, role=MessageRole.assistant
        ) 

        user_message_result = await message_collection().insert_one(
            user_message.model_dump()
        )  

        logger.info(f"[send_user_message] user_message_result: {user_message_result}")

        if user_message_result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to send message")

        request_message_result = await user_query_collection().insert_one(
            request_message.model_dump()
        )

        logger.info(f"[send_user_message] request_message_result: {request_message_result}")

        if request_message_result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to send message")

        response_message_result = await user_query_collection().insert_one(
            response_message.model_dump()
        ) 
        logger.info(f"[send_user_message] response_message_result: {response_message_result}")

        if response_message_result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to send message")

        response_message_result = await message_collection().insert_one(
            response_message.model_dump()
        )

        if response_message_result.inserted_id is None:
            raise HTTPException(status_code=500, detail="Failed to send message")

        response_message = await message_collection().find_one(
            {"_id": response_message_result.inserted_id}
        )
        request_message = await message_collection().find_one(
            {"_id": user_message_result.inserted_id}
        )

        return {
            "response_message": messageEntity(response_message),
            "request_message": messageEntity(request_message),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
    Update User Chat
"""
async def update_user_chat(chat_id: str, chat: UpdateUserChatRequest, user_id: str):
    try:
        existing_chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        if not existing_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        result = await chat_collection().update_one(
            {"_id": ObjectId(chat_id), "user_id": user_id},
            {"$set": {"name": chat.name}},
        )
        chat = await chat_collection().find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )
        return chatEntity(chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



# async def expand_query_with_context(chat_id: str, user_query: str) -> str:
#     try:
#         # Fetch messages ordered by creation time
#         messages = await message_collection().find(
#             {"chat_id": chat_id}
#         ).sort("created_at", 1).to_list(length=100)

#         # Convert messages into list of dicts with 'role' and 'content'
#         conversation = [
#             {"role": msg["role"], "content": msg["content"]}
#             for msg in messages
#             if msg.get("role") in ["user", "assistant"]
#         ]

#         # Log the conversation type and preview of content
#         logger.info(f"[expand_query_with_context] Conversation type: {type(conversation)}")
#         logger.debug(f"[expand_query_with_context] First 2 messages: {conversation[:2]}")

#         # Call the LLM to expand the user query
#         expanded_query = expand_user_query(conversation, user_query)
#         logger.info(f"[expand_query_with_context] Expanded query: {expanded_query}")

#         return expanded_query

#     except Exception as e:
#         logger.error(f"[expand_query_with_context] Failed to expand query: {str(e)}", exc_info=True)
#         return user_query  # fallback to original query if expansion fails
    

# import logging
# from app.utils.llm import metadata_query
# from app.utils.llm import expand_user_query  # Adjust import path if needed

# from app.db.mongodb import message_collection  # Adjust import path if needed
# async def expand_query_with_context(chat_id: str, user_query: str) -> str:
#     try:
#         # Fetch messages ordered by creation time
#         messages = await message_collection().find(
#             {"chat_id": chat_id}
#         ).sort("created_at", 1).to_list(length=100)
#         # Convert messages into list of dicts with 'role' and 'content'
#         conversation = [
#             {"role": msg["role"], "content": msg["content"]}
#             for msg in messages
#             if msg.get("role") in ["user", "assistant"]
#         ]
#         # Log the conversation type and preview of content
#         logger.info(f"[expand_query_with_context] Conversation type: {type(conversation)}")
#         logger.debug(f"[expand_query_with_context] First 2 messages: {conversation[:2]}")
#         # Call the LLM to expand the user query
#         expanded_query = expand_user_query(conversation, user_query) 
#         # metadata_query_result = metadata_query(expanded_query)
#         # logger.info(f"[expand_query_with_context] Metadata query result: {metadata_query_result}")
#         logger.info(f"[expand_query_with_context] Expanded query: {expanded_query}")
#         return expanded_query
#     except Exception as e:
#         logger.error(f"[expand_query_with_context] Failed to expand query: {str(e)}", exc_info=True)
#         return user_query  # fallback to original query if expansion fails