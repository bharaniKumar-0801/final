def chatEntity(chat) -> dict:
    return {
        "id": str(chat["_id"]),
        "name": chat["name"],
        "created_at": chat["created_at"],
        "updated_at": chat["updated_at"],
    }


def chatListEntity(chats) -> list:
    return [chatEntity(chat) for chat in chats]
