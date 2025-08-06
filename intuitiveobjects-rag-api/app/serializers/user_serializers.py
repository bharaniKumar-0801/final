def userEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
        "password": user["password"],
    }


def userResponseEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }


def userListEntity(users) -> list:
    return [userEntity(user) for user in users]


def userResponseListEntity(users) -> list:
    return [userResponseEntity(user) for user in users]
