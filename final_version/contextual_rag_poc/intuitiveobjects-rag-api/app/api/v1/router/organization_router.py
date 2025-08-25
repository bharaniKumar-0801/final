from fastapi import APIRouter, HTTPException
from app.schema.organization_schema import (
    CreateOrganizationSchema,
    UpdateOrganizationSchema,
)
import app.services.organization_services as organization_services

organization_router = APIRouter()


# Create Organization
@organization_router.post("/register")
async def create_organization(organization: CreateOrganizationSchema):
    try:
        organization = await organization_services.create_organization(organization)
        return {
            "message": "Organization created successfully",
            "success": True,
            "data": organization,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get All Organizations
@organization_router.get("/all")
async def get_all_organizations():
    organizations = await organization_services.get_all_organizations()
    return {
        "message": "All Organizations retrieved successfully",
        "success": True,
        "data": organizations,
    }


# Get Organization
@organization_router.get("/{organization_id}")
async def get_organization(organization_id: str):
    try:
        organization = await organization_services.get_organization(organization_id)
        return {
            "message": "Organization retrieved successfully",
            "success": True,
            "data": organization,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Delete Organization
@organization_router.delete("/{organization_id}")
async def delete_organization(organization_id: str):
    organization = await organization_services.delete_organization(organization_id)
    return {
        "message": "Organization deleted successfully",
        "success": True,
        "data": organization,
    }


# Update Organization
@organization_router.put("/{organization_id}")
async def update_organization(
    organization_id: str, organization: UpdateOrganizationSchema
):
    try:
        organization = await organization_services.update_organization(
            organization_id, organization
        )
        return {
            "message": "Organization updated successfully",
            "success": True,
            "data": organization,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
