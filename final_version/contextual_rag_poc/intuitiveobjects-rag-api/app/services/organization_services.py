from app.db.mongodb import organization_collection
from app.schema.organization_schema import (
    CreateOrganizationSchema,
    UpdateOrganizationSchema,
)
from fastapi import HTTPException
from app.models.organization_model import Organization
from app.serializers.organization_serializers import (
    OrganizationListEntity,
    OrganizationEntity,
)
from bson.objectid import ObjectId


async def create_organization(organization: CreateOrganizationSchema):
    existing_organization = await organization_collection().find_one(
        {"name": organization.name}
    )

    if existing_organization:
        raise HTTPException(status_code=400, detail="Organization already exists")

    organization_data = Organization(name=organization.name)
    result = await organization_collection().insert_one(organization_data.model_dump())

    new_organization = await organization_collection().find_one(
        {"_id": result.inserted_id}
    )

    return OrganizationEntity(new_organization)


async def get_organization(organization_id: str):
    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    return OrganizationEntity(existing_organization)


async def get_all_organizations():
    organizations = await organization_collection().find().to_list()
    return OrganizationListEntity(organizations)


async def delete_organization(organization_id):
    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    await organization_collection().delete_one({"_id": ObjectId(organization_id)})

    return OrganizationEntity(existing_organization)


async def update_organization(organization_id, organization: UpdateOrganizationSchema):
    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    await organization_collection().update_one(
        {"_id": ObjectId(organization_id)}, {"$set": {"name": organization.name}}
    )

    updated_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    return OrganizationEntity(updated_organization)
