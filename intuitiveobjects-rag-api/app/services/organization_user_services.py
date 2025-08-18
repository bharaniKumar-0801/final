from fastapi import HTTPException, status
from app.schema.organization_user_schema import (
    CreateOrganizationUserSchema,
    UpdateOrganizationUserSchema,
    LoginOrganizationUserSchema,
)
from app.models.organization_user_model import OrganizationUser
from app.models.token_model import Token
from app.db.mongodb import (
    organization_user_collection,
    category_collection,
    organization_admin_collection,
    token_collection,
)
from app.utils.auth import hash_password, verify_password, generate_token
from bson.objectid import ObjectId
from app.serializers.organization_user_serializers import (
    OrganizationUserEntity,
    OrganizationUserListEntity,
)


async def create_organization_user(
    organization_user: CreateOrganizationUserSchema, user_id: str
):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=400, detail="User not found")

    existing_organization_user = await organization_user_collection().find_one(
        {"email": organization_user.email}
    )
    if existing_organization_user:
        raise HTTPException(status_code=400, detail="Organization user already exists")

    existing_category = await category_collection().find_one(
        {"_id": ObjectId(organization_user.category_id)}
    )

    if not existing_category:
        raise HTTPException(status_code=400, detail="Category not found")

    organization_id = existing_category["organization_id"]
    hashed_password = hash_password(organization_user.password)

    organization_user_data = OrganizationUser(
        email=organization_user.email,
        password=hashed_password,
        category_id=organization_user.category_id,
        organization_id=organization_id,
    )

    result = await organization_user_collection().insert_one(
        organization_user_data.model_dump()
    )

    new_organization_user = await organization_user_collection().find_one(
        {"_id": result.inserted_id}
    )

    return OrganizationUserEntity(new_organization_user)


async def get_organization_users(user_id: str):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=400, detail="User not found")

    organization_id = existing_user["organization_id"]

    organization_users = (
        await organization_user_collection()
        .find({"organization_id": organization_id})
        .to_list(None)
    )
    return OrganizationUserListEntity(organization_users)


async def update_organization_user(
    organization_user_id: str,
    organization_user: UpdateOrganizationUserSchema,
    user_id: str,
):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=400, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization_user = await organization_user_collection().find_one(
        {"_id": ObjectId(organization_user_id), "organization_id": organization_id}
    )

    if not existing_organization_user:
        raise HTTPException(status_code=400, detail="Organization user not found")

    if organization_user.category_id:
        existing_category = await category_collection().find_one(
            {"_id": ObjectId(organization_user.category_id)}
        )
        if not existing_category:
            raise HTTPException(status_code=400, detail="Category not found")

        if (
            existing_organization_user["organization_id"]
            != existing_category["organization_id"]
        ):
            raise HTTPException(
                status_code=400,
                detail="Organization user and category organization id mismatch",
            )
        existing_organization_user["category_id"] = organization_user.category_id

    if organization_user.password:
        existing_organization_user["password"] = hash_password(
            organization_user.password
        )

    if organization_user.email:
        existing_email_organization_user = (
            await organization_user_collection().find_one(
                {
                    "email": organization_user.email,
                    "organization_id": organization_id,
                    "_id": {"$ne": ObjectId(organization_user_id)},
                }
            )
        )

        if existing_email_organization_user:
            raise HTTPException(
                status_code=400, detail="Organization user already exists"
            )
        existing_organization_user["email"] = organization_user.email
    organization_user_data = OrganizationUser(
        email=existing_organization_user["email"],
        password=existing_organization_user["password"],
        category_id=existing_organization_user["category_id"],
        organization_id=existing_organization_user["organization_id"],
    )
    organization_user = await organization_user_collection().update_one(
        {"_id": ObjectId(organization_user_id)},
        {"$set": organization_user_data.model_dump()},
    )
    updated_organization_user = await organization_user_collection().find_one(
        {"_id": ObjectId(organization_user_id)}
    )
    return OrganizationUserEntity(updated_organization_user)


async def delete_organization_user(organization_user_id: str, user_id: str):
    existing_user = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_user:
        raise HTTPException(status_code=400, detail="User not found")

    organization_id = existing_user["organization_id"]

    existing_organization_user = await organization_user_collection().find_one(
        {"_id": ObjectId(organization_user_id), "organization_id": organization_id}
    )
    if not existing_organization_user:
        raise HTTPException(status_code=400, detail="Organization user not found")

    await organization_user_collection().delete_one(
        {"_id": ObjectId(organization_user_id)}
    )

    return OrganizationUserEntity(existing_organization_user)


async def login_organization_user(organization_user: LoginOrganizationUserSchema):
    existing_organization_user = await organization_user_collection().find_one(
        {"email": organization_user.email}
    )
    if not existing_organization_user:
        raise HTTPException(status_code=400, detail="Organization user not found")

    if not verify_password(
        organization_user.password, existing_organization_user["password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password"
        )

    user_id = str(existing_organization_user["_id"])

    token = generate_token(user_id)

    token_data = Token(user_id=user_id, token=token)

    await token_collection().delete_one({"user_id": user_id})
    await token_collection().insert_one(token_data.model_dump())

    return {"user": OrganizationUserEntity(existing_organization_user), "token": token}
