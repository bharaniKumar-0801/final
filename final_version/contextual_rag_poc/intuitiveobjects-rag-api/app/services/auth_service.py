from app.schema.user_schema import SignupSchema, SigninSchema
from app.models.user_model import User
from app.db.mongodb import user_collection, token_collection
from fastapi import HTTPException
from app.utils.auth import hash_password, generate_token, verify_password
from app.serializers.user_serializers import userResponseEntity
from app.models.token_model import Token


async def signup(user: SignupSchema):
    existing_user = await user_collection().find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.password)
    user_data = User(email=user.email, password=hashed_password)

    result = await user_collection().insert_one(user_data.model_dump())
    result_id = str(result.inserted_id)
    token = generate_token(result_id)

    token_data = Token(user_id=result_id, token=token)

    await token_collection().insert_one(token_data.model_dump())

    new_user = userResponseEntity(
        await user_collection().find_one({"_id": result.inserted_id})
    )
    return {"user": new_user, "token": token}


async def signin(user: SigninSchema):
    existing_user = await user_collection().find_one({"email": user.email})
    if not existing_user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(user.password, existing_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid password")

    user_id = str(existing_user["_id"])

    token = generate_token(user_id)

    token_data = Token(user_id=user_id, token=token)
    await token_collection().delete_one({"user_id": user_id})
    await token_collection().insert_one(token_data.model_dump())

    return {"user": userResponseEntity(existing_user), "token": token}
