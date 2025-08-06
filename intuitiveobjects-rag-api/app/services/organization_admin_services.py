from app.db.mongodb import (
    organization_admin_collection,
    organization_collection,
    token_collection,
    category_collection,
)
from app.schema.organization_admin_schema import (
    CreateOrganizationAdminSchema,
    UpdateOrganizationAdminSchema,
    LoginOrganizationAdminSchema,
    CreateCategorySchema,
    UpdateCategorySchema,
    CreateOrganizationAppConfigSchema,
)
from app.utils.auth import hash_password, verify_password, generate_token
from app.models.token_model import Token
from app.models.organization_admin_model import OrganizationAdmin, Category, OrganizationAppConfig
from app.serializers.organization_admin_serializers import (
    OrganizationAdminEntity,
    OrganizationAdminListEntity,
    CategoryEntity,
    CategoryListEntity,
    OrganizationAppConfigEntity,
    OrganizationAppConfigListEntity,
)
from bson.objectid import ObjectId
from fastapi import HTTPException
from app.db.mongodb import organization_app_config_collection

async def create_organization_admin(
    organization_id: str, organization_admin: CreateOrganizationAdminSchema
):
    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    existing_organization_admin = await organization_admin_collection().find_one(
        {"email": organization_admin.email}
    )

    if existing_organization_admin:
        raise HTTPException(status_code=400, detail="Organization admin already exists")

    hashed_password = hash_password(organization_admin.password)

    organization_admin_data = OrganizationAdmin(
        email=organization_admin.email,
        name=organization_admin.name,
        password=hashed_password,
        organization_id=organization_id,
        role="organization_admin",
    )

    result = await organization_admin_collection().insert_one(
        organization_admin_data.model_dump()
    )

    new_organization_admin = await organization_admin_collection().find_one(
        {"_id": result.inserted_id}
    )

    return OrganizationAdminEntity(new_organization_admin)


async def get_organization_admins(organization_id: str):
    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    existing_organization_admins = (
        await organization_admin_collection()
        .find(
            {
                "organization_id": organization_id,
            }
        )
        .to_list()
    )

    if not existing_organization_admins:
        raise HTTPException(status_code=404, detail="Organization admins not found")

    return OrganizationAdminListEntity(existing_organization_admins)


async def update_organization_admin(
    organization_admin_id: str, organization_admin: UpdateOrganizationAdminSchema
):
    existing_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(organization_admin_id)}
    )

    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    if organization_admin.email:
        existing_organization_admin["email"] = organization_admin.email

    if organization_admin.name:
        existing_organization_admin["name"] = organization_admin.name

    if organization_admin.password:
        existing_organization_admin["password"] = hash_password(
            organization_admin.password
        )

    await organization_admin_collection().update_one(
        {"_id": ObjectId(organization_admin_id)}, {"$set": existing_organization_admin}
    )

    updated_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(organization_admin_id)}
    )

    return OrganizationAdminEntity(updated_organization_admin)


async def delete_organization_admin(organization_admin_id: str):
    existing_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(organization_admin_id)}
    )

    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    await organization_admin_collection().delete_one(
        {"_id": ObjectId(organization_admin_id)}
    )

    return OrganizationAdminEntity(existing_organization_admin)


async def login_organization_admin(organization_admin: LoginOrganizationAdminSchema):
    existing_organization_admin = await organization_admin_collection().find_one(
        {"email": organization_admin.email}
    )

    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    if not verify_password(
        organization_admin.password, existing_organization_admin["password"]
    ):
        raise HTTPException(status_code=401, detail="Invalid password")

    user_id = str(existing_organization_admin["_id"])

    token = generate_token(user_id)

    token_data = Token(user_id=user_id, token=token)

    await token_collection().delete_one({"user_id": user_id})
    await token_collection().insert_one(token_data.model_dump())

    return {
        "user": OrganizationAdminEntity(existing_organization_admin),
        "token": token,
    }


async def create_category(category: CreateCategorySchema, user_id: str):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    existing_category = await category_collection().find_one(
        {"name": category.name, "organization_id": organization_id}
    )

    if existing_category:
        raise HTTPException(status_code=400, detail="Category already exists")

    new_category = Category(name=category.name, organization_id=organization_id)

    result = await category_collection().insert_one(new_category.model_dump())

    new_category = await category_collection().find_one({"_id": result.inserted_id})

    return CategoryEntity(new_category)


async def get_categories(user_id: str):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    categories = (
        await category_collection().find({"organization_id": organization_id}).to_list()
    )

    return CategoryListEntity(categories)


async def update_category(
    category_id: str, category: UpdateCategorySchema, user_id: str
):
    existing_category = await category_collection().find_one(
        {"_id": ObjectId(category_id)}
    )

    if not existing_category:
        raise HTTPException(status_code=404, detail="Category not found")

    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    if existing_category["organization_id"] != organization_id:
        raise HTTPException(
            status_code=403, detail="You are not authorized to update this category"
        )

    updated_category = Category(name=category.name, organization_id=organization_id)

    await category_collection().update_one(
        {"_id": ObjectId(category_id)}, {"$set": updated_category.model_dump()}
    )

    updated_category = await category_collection().find_one(
        {"_id": ObjectId(category_id)}
    )

    return CategoryEntity(updated_category)


async def delete_category(category_id: str, user_id: str):
    existing_category = await category_collection().find_one(
        {"_id": ObjectId(category_id)}
    )

    if not existing_category:
        raise HTTPException(status_code=404, detail="Category not found")

    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    if existing_category["organization_id"] != organization_id:
        raise HTTPException(
            status_code=403, detail="You are not authorized to delete this category"
        )

    await category_collection().delete_one({"_id": ObjectId(category_id)})

    return CategoryEntity(existing_category)


# async def create_organization_app_config(
#     organization_app_config: CreateOrganizationAppConfigSchema, user_id: str
# ):
#     existing_user = await organization_admin_collection().find_one(
#         {"_id": ObjectId(user_id)}
#     )

#     if not existing_user:
#         raise HTTPException(status_code=404, detail="User not found")

#     organization_id = existing_user["organization_id"]

#     existing_organization = await organization_collection().find_one(
#         {"_id": ObjectId(organization_id)}
#     )

#     if not existing_organization:
#         raise HTTPException(status_code=404, detail="Organization not found")

#     new_organization_app_config = OrganizationAppConfig(
#         llm_model=organization_app_config.llm_model,
#         embedding_model=organization_app_config.embedding_model,
#         temperature=organization_app_config.temperature,
#         system_prompt=organization_app_config.system_prompt,
#         organization_id=organization_id,
#     )

#     result = await organization_app_config_collection().insert_one(new_organization_app_config.model_dump())

#     new_organization_app_config = await organization_app_config_collection().find_one({"_id": result.inserted_id})

#     return OrganizationAppConfigEntity(new_organization_app_config)

from datetime import datetime
async def create_organization_app_config(
    organization_app_config: CreateOrganizationAppConfigSchema, user_id: str
):
    # Step 1: Check if user exists
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Step 2: Get organization_id from user
    organization_id = existing_user["organization_id"]

    # Step 3: Check if organization exists
    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )
    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Step 4: Check if config already exists
    existing_config = await organization_app_config_collection().find_one(
        {"organization_id": organization_id}
    )

    config_data = {
        "llm_model": organization_app_config.llm_model,
        "embedding_model": organization_app_config.embedding_model,
        "temperature": organization_app_config.temperature,
        "system_prompt": organization_app_config.system_prompt,
        "organization_id": organization_id,
        "updated_at": datetime.utcnow(),
    }

    if existing_config:
        # Update existing config
        await organization_app_config_collection().update_one(
            {"_id": existing_config["_id"]},
            {"$set": config_data}
        )
        updated_config = await organization_app_config_collection().find_one(
            {"_id": existing_config["_id"]}
        )
        return OrganizationAppConfigEntity(updated_config)
    else:
        # Create new config
        config_data["created_at"] = datetime.utcnow()
        result = await organization_app_config_collection().insert_one(config_data)
        new_config = await organization_app_config_collection().find_one(
            {"_id": result.inserted_id}
        )
        return OrganizationAppConfigEntity(new_config)


async def get_organization_app_configs(user_id: str):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization = await organization_collection().find_one(
        {"_id": ObjectId(organization_id)}
    )

    if not existing_organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Get the latest (most recent) app config for the organization
    latest_config = await organization_app_config_collection().find_one(
        {"organization_id": organization_id}, sort=[("created_at", -1)]
    )
    if not latest_config:
        return []
    return OrganizationAppConfigListEntity([latest_config])