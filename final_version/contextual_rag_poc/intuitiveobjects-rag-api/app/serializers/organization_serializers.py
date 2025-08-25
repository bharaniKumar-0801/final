def OrganizationEntity(organization) -> dict:
    return {
        "id": str(organization["_id"]),
        "name": organization["name"],
        "created_at": organization["created_at"],
        "updated_at": organization["updated_at"],
    }


def OrganizationListEntity(organizations) -> list:
    return [OrganizationEntity(organization) for organization in organizations]
