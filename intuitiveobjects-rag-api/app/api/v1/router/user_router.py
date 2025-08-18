from fastapi import APIRouter, Depends, HTTPException
from app.schema.user_schema import SignupSchema, SigninSchema
import app.services.auth_service as auth_service
from app.db.mongodb import connect_to_mongodb

user_router = APIRouter()

@user_router.post("/signup")
async def signup(user: SignupSchema):
    await connect_to_mongodb()
    result = await auth_service.signup(user)
    return {"message": "User created successfully", "success": True, "data": result}


@user_router.post("/signin")
async def signin(user: SigninSchema):
    await connect_to_mongodb()
    result = await auth_service.signin(user)
    return {"message": "User logged in successfully", "success": True, "data": result}
