def messageEntity(message) -> dict:
    return {
        "id": str(message["_id"]),
        "content": message["content"],
        "role": message["role"],
        "created_at": message["created_at"],
        "updated_at": message["updated_at"],
    }


def messageListEntity(messages) -> list:
    return [messageEntity(message) for message in messages]
