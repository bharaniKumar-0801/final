from fastapi import APIRouter, HTTPException, Depends
import app.schema.organization_admin_schema as organization_admin_schema
import app.services.organization_admin_services as organization_admin_services
from app.utils.auth import get_current_user

organization_admin_router = APIRouter()


@organization_admin_router.post("/organization/{organization_id}")
async def create_organization_admin(
    organization_id: str,
    organization_admin: organization_admin_schema.CreateOrganizationAdminSchema,
):
    try:
        organization_admin = (
            await organization_admin_services.create_organization_admin(
                organization_id, organization_admin
            )
        )
        return {
            "message": "Organization admin created successfully",
            "success": True,
            "data": organization_admin,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.get("/organization/{organization_id}")
async def get_organization_admin(organization_id: str):
    try:
        organization_admin = await organization_admin_services.get_organization_admins(
            organization_id
        )
        return {
            "message": "Organization admin retrieved successfully",
            "success": True,
            "data": organization_admin,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.put("/{organization_admin_id}")
async def update_organization_admin(
    organization_admin_id: str,
    organization_admin: organization_admin_schema.UpdateOrganizationAdminSchema,
):
    try:
        organization_admin = (
            await organization_admin_services.update_organization_admin(
                organization_admin_id, organization_admin
            )
        )
        return {
            "message": "Organization admin updated successfully",
            "success": True,
            "data": organization_admin,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.delete("/{organization_admin_id}")
async def delete_organization_admin(organization_admin_id: str):
    try:
        organization_admin = (
            await organization_admin_services.delete_organization_admin(
                organization_admin_id
            )
        )
        return {
            "message": "Organization admin deleted successfully",
            "success": True,
            "data": organization_admin,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.post("/login")
async def login_organization_admin(
    organization_admin: organization_admin_schema.LoginOrganizationAdminSchema,
):
    try:
        organization_admin = await organization_admin_services.login_organization_admin(
            organization_admin
        )
        return {
            "message": "Organization admin logged in successfully",
            "success": True,
            "data": organization_admin,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.post("/category")
async def create_category(
    category: organization_admin_schema.CreateCategorySchema,
    user_id: str = Depends(get_current_user),
):
    try:
        category = await organization_admin_services.create_category(category, user_id)
        return {
            "message": "Category created successfully",
            "success": True,
            "data": category,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.get("/category")
async def get_categories(user_id: str = Depends(get_current_user)):
    try:
        categories = await organization_admin_services.get_categories(user_id)
        return {
            "message": "Categories retrieved successfully",
            "success": True,
            "data": categories,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.put("/category/{category_id}")
async def update_category(
    category_id: str,
    category: organization_admin_schema.UpdateCategorySchema,
    user_id: str = Depends(get_current_user),
):
    try:
        category = await organization_admin_services.update_category(
            category_id, category, user_id
        )
        return {
            "message": "Category updated successfully",
            "success": True,
            "data": category,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.delete("/category/{category_id}")
async def delete_category(category_id: str, user_id: str = Depends(get_current_user)):
    try:
        category = await organization_admin_services.delete_category(
            category_id, user_id
        )
        return {
            "message": "Category deleted successfully",
            "success": True,
            "data": category,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@organization_admin_router.post("/organization-app-config")
async def create_organization_app_config(
    organization_app_config: organization_admin_schema.CreateOrganizationAppConfigSchema,
    user_id: str = Depends(get_current_user),
):
    try:
        organization_app_config = (
            await organization_admin_services.create_organization_app_config(
                organization_app_config, user_id
            )
        )
        return {
            "message": "Organization app config created successfully",
            "success": True,
            "data": organization_app_config,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@organization_admin_router.get("/organization-app-config")
async def get_organization_app_config(user_id: str = Depends(get_current_user)):
    try:
        organization_app_config = await organization_admin_services.get_organization_app_configs(user_id)
        return {
            "message": "Organization app config retrieved successfully",
            "success": True,
            "data": organization_app_config,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
