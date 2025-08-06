from fastapi import APIRouter, HTTPException, Depends
from app.schema.organization_user_schema import (
    CreateOrganizationUserSchema,
    UpdateOrganizationUserSchema,
    LoginOrganizationUserSchema,
)
import app.services.organization_user_services as organization_user_services
from app.utils.auth import get_current_user

organization_user_router = APIRouter()


@organization_user_router.post("/")
async def create_organization_user(
    organization_user: CreateOrganizationUserSchema,
    user_id: str = Depends(get_current_user),
):
    try:
        organization_user = await organization_user_services.create_organization_user(
            organization_user, user_id
        )
        return {
            "message": "Organization user created successfully",
            "success": True,
            "organization_user": organization_user,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_user_router.get("/")
async def get_organization_users(user_id: str = Depends(get_current_user)):
    try:
        organization_users = await organization_user_services.get_organization_users(
            user_id
        )
        return {
            "message": "Organization users fetched successfully",
            "success": True,
            "organization_users": organization_users,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_user_router.put("/{organization_user_id}")
async def update_organization_user(
    organization_user_id: str,
    organization_user: UpdateOrganizationUserSchema,
    user_id: str = Depends(get_current_user),
):
    try:
        organization_user = await organization_user_services.update_organization_user(
            organization_user_id, organization_user, user_id
        )
        return {
            "message": "Organization user updated successfully",
            "success": True,
            "organization_user": organization_user,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_user_router.delete("/{organization_user_id}")
async def delete_organization_user(
    organization_user_id: str, user_id: str = Depends(get_current_user)
):
    try:
        organization_user = await organization_user_services.delete_organization_user(
            organization_user_id, user_id
        )
        return {
            "message": "Organization user deleted successfully",
            "success": True,
            "data": organization_user,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_user_router.post("/login")
async def login_organization_user(organization_user: LoginOrganizationUserSchema):
    try:
        organization_user = await organization_user_services.login_organization_user(
            organization_user
        )
        return {
            "message": "Organization user logged in successfully",
            "success": True,
            "organization_user": organization_user,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
