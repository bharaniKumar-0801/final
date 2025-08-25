def OrganizationUserEntity(organization_user) -> dict:
    return {
        "id": str(organization_user["_id"]),
        "email": organization_user["email"],
        "category_id": organization_user["category_id"],
        "organization_id": organization_user["organization_id"],
        "is_active": organization_user["is_active"],
        "created_at": organization_user["created_at"],
        "updated_at": organization_user["updated_at"],
    }


def OrganizationUserListEntity(organization_users) -> list:
    return [
        OrganizationUserEntity(organization_user)
        for organization_user in organization_users
    ]
