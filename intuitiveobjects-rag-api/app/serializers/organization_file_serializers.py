def OrganizationFileEntity(file) -> dict:
    return {
        "id": str(file["_id"]),
        "organization_id": file["organization_id"],
        "category_id": file["category_id"],
        "file_name": file["file_name"],
        "file_type": file["file_type"],
        "file_size": file["file_size"],
        "tags": file["tags"],
        "created_at": file["created_at"],
        "updated_at": file["updated_at"],
    }


def OrganizationFileListEntity(files) -> list:
    return [OrganizationFileEntity(file) for file in files]
