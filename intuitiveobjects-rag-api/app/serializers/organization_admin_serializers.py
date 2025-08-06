from app.models.organization_admin_model import (
    OrganizationAdmin,
    Category,
    OrganizationAppConfig,
)


def OrganizationAdminEntity(organization_admin: OrganizationAdmin) -> dict:
    return {
        "id": str(organization_admin["_id"]),
        "name": organization_admin["name"],
        "email": organization_admin["email"],
        "organization_id": organization_admin["organization_id"],
        "role": organization_admin["role"],
        "is_active": organization_admin["is_active"],
        "created_at": organization_admin["created_at"],
        "updated_at": organization_admin["updated_at"],
    }


def OrganizationAdminListEntity(organization_admins) -> list:
    return [
        OrganizationAdminEntity(organization_admin)
        for organization_admin in organization_admins
    ]


def CategoryEntity(category: Category) -> dict:
    return {
        "id": str(category["_id"]),
        "name": category["name"],
        "created_at": category["created_at"],
        "updated_at": category["updated_at"],
    }


def CategoryListEntity(categories) -> list:
    return [CategoryEntity(category) for category in categories]


def OrganizationAppConfigEntity(organization_app_config: OrganizationAppConfig) -> dict:
    return {
        "id": str(organization_app_config["_id"]),
        "llm_model": organization_app_config["llm_model"],
        "embedding_model": organization_app_config["embedding_model"],
        "temperature": organization_app_config["temperature"],
        "system_prompt": organization_app_config["system_prompt"],
        "organization_id": organization_app_config["organization_id"],
        "created_at": organization_app_config["created_at"],
        "updated_at": organization_app_config["updated_at"],
    }


def OrganizationAppConfigListEntity(organization_app_configs) -> list:
    return [
        OrganizationAppConfigEntity(organization_app_config)
        for organization_app_config in organization_app_configs
    ]
