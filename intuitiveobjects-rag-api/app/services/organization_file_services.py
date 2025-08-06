# from app.models.organization_file_model import OrganizationFile
# from fastapi import UploadFile, HTTPException
# from app.db.mongodb import (
#     organization_admin_collection,
#     organization_file_collection,
#     category_collection,
#     get_fs
# )
# from bson import ObjectId
# import os
# import io
# from app.core.config import settings
# from app.utils.helpers import ensure_directory_exists
# from app.serializers.organization_file_serializers import (
#     OrganizationFileEntity,
#     OrganizationFileListEntity,
# )
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from googleapiclient.http import MediaIoBaseUpload
# from google.oauth2 import service_account
# from typing import List
# from app.utils.zip_utils import process_zip_file_return_contents



# # async def organization_upload_file(
# #     category_id: str, file: UploadFile, tags: List[str], user_id: str
# # ):
# #     existing_organization_admin = await organization_admin_collection().find_one(
# #         {"_id": ObjectId(user_id)}
# #     )

# #     if not existing_organization_admin:
# #         raise HTTPException(status_code=404, detail="Organization admin not found")


# #     file_name = file.filename
# #     file_size = file.size
# #     file_type = file.content_type
    
# #     # Save the file content to GridFS
# #     grid_in = await get_fs().upload_from_stream(
# #         file.filename,
# #         file.file,
# #         metadata={"contentType": file_type, "organizationId": existing_organization_admin["organization_id"]}
# #     )

# #     file_data = OrganizationFile(
# #         organization_id=existing_organization_admin["organization_id"],
# #         category_id=category_id,
# #         file_name=file_name,
# #         file_type=file_type,
# #         file_size=file_size,
# #         tags=tags,
# #         storage_id=str(grid_in)  # Save the GridFS file ID
# #     )


# #     result = await organization_file_collection().insert_one(file_data.model_dump())

# #     new_file = await organization_file_collection().find_one(
# #         {"_id": result.inserted_id}
# #     )

# #     return OrganizationFileEntity(new_file)
# # Bharani Kumar
# async def organization_upload_file(
#     category_id: str, file: UploadFile, tags: List[str], user_id: str
# ):
#     # Validate user
#     existing_organization_admin = await organization_admin_collection().find_one(
#         {"_id": ObjectId(user_id)}
#     )
#     if not existing_organization_admin:
#         raise HTTPException(status_code=404, detail="Organization admin not found")

#     file_name = file.filename
#     file_type = file.content_type

#     # Read file content to determine size
#     file_content = await file.read()
#     file_size = len(file_content)
#     file.file.seek(0)  # Important: rewind for GridFS

#     # Upload to GridFS (ensure get_fs returns AsyncIOMotorGridFSBucket)
#     grid_in = await get_fs().upload_from_stream(
#         file_name,
#         file.file,
#         metadata={
#             "contentType": file_type,
#             "organizationId": existing_organization_admin["organization_id"]
#         }
#     )

#     # Save metadata to your collection
#     file_data = OrganizationFile(
#         organization_id=existing_organization_admin["organization_id"],
#         category_id=category_id,
#         file_name=file_name,
#         file_type=file_type,
#         file_size=file_size,
#         tags=tags,
#         storage_id=str(grid_in)  # GridFS file ID
#     )

#     result = await organization_file_collection().insert_one(file_data.model_dump())
#     new_file = await organization_file_collection().find_one({"_id": result.inserted_id})

#     return OrganizationFileEntity(new_file)



# # async def organization_upload_file(
# #     category_id: str, file: UploadFile, tags: List[str], user_id: str
# # ):
# #     # Validate user
# #     existing_organization_admin = await organization_admin_collection().find_one(
# #         {"_id": ObjectId(user_id)}
# #     )
# #     if not existing_organization_admin:
# #         raise HTTPException(status_code=404, detail="Organization admin not found")

# #     file_name = file.filename
# #     file_type = file.content_type
# #     file_content = await file.read()
# #     file_size = len(file_content)
# #     file.file.seek(0)

# #     # Upload file to GridFS (original file)
# #     grid_in = await get_fs().upload_from_stream(
# #         file_name,
# #         file.file,
# #         metadata={
# #             "contentType": file_type,
# #             "organizationId": existing_organization_admin["organization_id"]
# #         }
# #     )

# #     if file_type == "application/zip":
# #         zip_hash, file_contents = process_zip_file_return_contents(file_content)

# #         # Save each file inside the ZIP to the same collection
# #         for doc in file_contents:
# #             await organization_file_collection().update_one(
# #                 {"_id": doc["_id"]},
# #                 {"$set": doc},
# #                 upsert=True
# #             )

# #         # Save metadata for the ZIP archive itself
# #         await organization_file_collection().update_one(
# #             {"_id": zip_hash},
# #             {"$set": {
# #                 "type": "zip",
# #                 "files": [d["_id"] for d in file_contents],
# #                 "status": "uploaded",
# #                 "filename": file_name,
# #                 "organization_id": existing_organization_admin["organization_id"],
# #                 "category_id": category_id,
# #                 "file_size": file_size,
# #                 "tags": tags,
# #                 "storage_id": str(grid_in),
# #                 "file_type": file_type
# #             }},
# #             upsert=True
# #         )

# #         new_file = await organization_file_collection().find_one({"_id": zip_hash})
# #         return OrganizationFileEntity(new_file)

# #     # Non-ZIP file fallback (optional)
# #     file_data = OrganizationFile(
# #         organization_id=existing_organization_admin["organization_id"],
# #         category_id=category_id,
# #         file_name=file_name,
# #         file_type=file_type,
# #         file_size=file_size,
# #         tags=tags,
# #         storage_id=str(grid_in)
# #     )

# #     result = await organization_file_collection().insert_one(file_data.model_dump())
# #     new_file = await organization_file_collection().find_one({"_id": result.inserted_id})

# #     return OrganizationFileEntity(new_file)



# async def organization_get_files(user_id: str):
#     existing_organization_admin = await organization_admin_collection().find_one(
#         {"_id": ObjectId(user_id)}
#     )

#     if not existing_organization_admin:
#         raise HTTPException(status_code=404, detail="Organization admin not found")

#     files = (
#         await organization_file_collection()
#         .find({"organization_id": existing_organization_admin["organization_id"]})
#         .to_list(None)
#     )

#     return OrganizationFileListEntity(files)


# async def organization_delete_file(file_id: str, user_id: str):
#     existing_organization_admin = await organization_admin_collection().find_one(
#         {"_id": ObjectId(user_id)}
#     )

#     if not existing_organization_admin:
#         raise HTTPException(status_code=404, detail="Organization admin not found")

#     print(file_id)
#     existing_file = await organization_file_collection().find_one(
#         {
#             "_id": ObjectId(file_id),
#             "organization_id": existing_organization_admin["organization_id"],
#         }
#     )

#     if not existing_file:
#         raise HTTPException(status_code=404, detail="File not found")

#     await organization_file_collection().delete_one({"_id": ObjectId(file_id)})

#     return OrganizationFileEntity(existing_file)


# async def organization_google_drive_upload_file(
#     category_id: str, file: UploadFile, user_id: str
# ):
#     # Verify the user is an organization admin
#     existing_organization_admin = await organization_admin_collection().find_one(
#         {"_id": ObjectId(user_id)}
#     )

#     if not existing_organization_admin:
#         raise HTTPException(status_code=404, detail="Organization admin not found")

#     # Get the category
#     category = await category_collection().find_one({"_id": ObjectId(category_id)})
#     if not category:
#         raise HTTPException(status_code=404, detail="Category not found")

#     # Initialize Google Drive API client
#     creds = None
#     # Load credentials from your preferred method (service account, OAuth, etc.)
#     SERVICE_ACCOUNT_FILE = os.path.join(settings.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE)
#     SCOPES = ["https://www.googleapis.com/auth/drive"]

#     try:
#         creds = service_account.Credentials.from_service_account_file(
#             SERVICE_ACCOUNT_FILE, scopes=SCOPES
#         )
#         drive_service = build("drive", "v3", credentials=creds)

#         # # Check if category has a Google Drive folder ID, if not create one
#         # drive_folder_id = category.get("drive_folder_id")
#         # print(drive_folder_id)

#         # if not drive_folder_id:
#         #     # Create a folder for this category
#         #     folder_metadata = {
#         #         "name": f"Category_{category.get('name', category_id)}",
#         #         "mimeType": "application/vnd.google-apps.folder",
#         #         "parents": ["1XlxMc3tugNMSWAbrnbuziHBadBunPysn"],
#         #     }

#         #     folder = (
#         #         drive_service.files()
#         #         .create(body=folder_metadata, fields="id")
#         #         .execute()
#         #     )

#         #     drive_folder_id = folder.get("id")

#         #     # Update category with the new folder ID
#         #     await category_collection().update_one(
#         #         {"_id": ObjectId(category_id)},
#         #         {"$set": {"drive_folder_id": drive_folder_id}},
#         #     )

#         # Read file content
#         file_content = await file.read()
#         file_name = file.filename
#         file_size = len(file_content)
#         file_type = file.content_type

#         # Create file metadata for Google Drive
#         file_metadata = {
#             "name": file_name,
#             "parents": ["1XlxMc3tugNMSWAbrnbuziHBadBunPysn"],  # Using the Drive folder ID associated with the category
#         }

#         # Create a BytesIO object from the file content
#         media = MediaIoBaseUpload(
#             io.BytesIO(file_content), mimetype=file_type, resumable=True
#         )

#         # Upload file to Google Drive
#         uploaded_file = (
#             drive_service.files()
#             .create(
#                 body=file_metadata,
#                 media_body=media,
#                 fields="id,name,mimeType,size,webViewLink",
#             )
#             .execute()
#         )

#         # Store file information in your database
#         file_data = OrganizationFile(
#             organization_id=existing_organization_admin["organization_id"],
#             category_id=category_id,
#             file_name=file_name,
#             file_type=file_type,
#             file_size=file_size,
#             type="google_drive",
#             drive_file_id=uploaded_file.get("id"),
#             drive_web_link=uploaded_file.get("webViewLink"),
#         )

#         result = await organization_file_collection().insert_one(file_data.model_dump())
#         new_file = await organization_file_collection().find_one(
#             {"_id": result.inserted_id}
#         )

#         return OrganizationFileEntity(new_file)

#     except HttpError as error:
#         print(error)
#         raise HTTPException(status_code=500, detail=f"Google Drive API error: {error}")
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")



from app.models.organization_file_model import OrganizationFile
from fastapi import UploadFile, HTTPException
from app.db.mongodb import (
    organization_admin_collection,
    organization_file_collection,
    category_collection,
    get_fs
)
from bson import ObjectId
import os
import io
from app.core.config import settings
from app.utils.helpers import ensure_directory_exists
from app.serializers.organization_file_serializers import (
    OrganizationFileEntity,
    OrganizationFileListEntity,
)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account
from typing import List



async def organization_upload_file(
    category_id: str, file: UploadFile, tags: List[str], user_id: str ,file_id :str
):
    # Validate user
    existing_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )
    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    file_name = file.filename
    file_type = file.content_type

    # Read file content to determine size
    file_content = await file.read()
    file_size = len(file_content)
    file.file.seek(0)  # Important: rewind for GridFS

    # Upload to GridFS (ensure get_fs returns AsyncIOMotorGridFSBucket)
    # grid_in = await get_fs().upload_from_stream(
    #     file_name,
    #     file.file,
    #     metadata={
    #         "contentType": file_type,
    #         "organizationId": existing_organization_admin["organization_id"]
    #     }
    # )

    # Save metadata to your collection
    file_data = OrganizationFile(
        id = file_id ,
        organization_id=existing_organization_admin["organization_id"],
        category_id=category_id,
        file_name=file_name,
        file_type=file_type,
        file_size=file_size,
        tags=tags,
        # organizationId=existing_organization_admin["organization_id"],
        # storage_id=str(grid_in)  # GridFS file ID
    )

    result = await organization_file_collection().insert_one(file_data.model_dump())
    new_file = await organization_file_collection().find_one({"_id": result.inserted_id})

    return OrganizationFileEntity(new_file)



async def organization_get_files(user_id: str):
    existing_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    files = (
        await organization_file_collection()
        .find({"organization_id": existing_organization_admin["organization_id"]})
        .to_list(None)
    )

    return OrganizationFileListEntity(files)


async def organization_delete_file(file_id: str, user_id: str):
    existing_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    print(file_id)
    existing_file = await organization_file_collection().find_one(
        {
            "_id": ObjectId(file_id),
            "organization_id": existing_organization_admin["organization_id"],
        }
    )

    if not existing_file:
        raise HTTPException(status_code=404, detail="File not found")

    await organization_file_collection().delete_one({"_id": ObjectId(file_id)})

    return OrganizationFileEntity(existing_file)


async def organization_google_drive_upload_file(
    category_id: str, file: UploadFile, user_id: str
):
    # Verify the user is an organization admin
    existing_organization_admin = await organization_admin_collection().find_one(
        {"_id": ObjectId(user_id)}
    )

    if not existing_organization_admin:
        raise HTTPException(status_code=404, detail="Organization admin not found")

    # Get the category
    category = await category_collection().find_one({"_id": ObjectId(category_id)})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Initialize Google Drive API client
    creds = None
    # Load credentials from your preferred method (service account, OAuth, etc.)
    SERVICE_ACCOUNT_FILE = os.path.join(settings.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE)
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        drive_service = build("drive", "v3", credentials=creds)

        # # Check if category has a Google Drive folder ID, if not create one
        # drive_folder_id = category.get("drive_folder_id")
        # print(drive_folder_id)

        # if not drive_folder_id:
        #     # Create a folder for this category
        #     folder_metadata = {
        #         "name": f"Category_{category.get('name', category_id)}",
        #         "mimeType": "application/vnd.google-apps.folder",
        #         "parents": ["1XlxMc3tugNMSWAbrnbuziHBadBunPysn"],
        #     }

        #     folder = (
        #         drive_service.files()
        #         .create(body=folder_metadata, fields="id")
        #         .execute()
        #     )

        #     drive_folder_id = folder.get("id")

        #     # Update category with the new folder ID
        #     await category_collection().update_one(
        #         {"_id": ObjectId(category_id)},
        #         {"$set": {"drive_folder_id": drive_folder_id}},
        #     )

        # Read file content
        file_content = await file.read()
        file_name = file.filename
        file_size = len(file_content)
        file_type = file.content_type

        # Create file metadata for Google Drive
        file_metadata = {
            "name": file_name,
            "parents": ["1XlxMc3tugNMSWAbrnbuziHBadBunPysn"],  # Using the Drive folder ID associated with the category
        }

        # Create a BytesIO object from the file content
        media = MediaIoBaseUpload(
            io.BytesIO(file_content), mimetype=file_type, resumable=True
        )

        # Upload file to Google Drive
        uploaded_file = (
            drive_service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id,name,mimeType,size,webViewLink",
            )
            .execute()
        )

        # Store file information in your database
        file_data = OrganizationFile(
            organization_id=existing_organization_admin["organization_id"],
            category_id=category_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            type="google_drive",
            drive_file_id=uploaded_file.get("id"),
            drive_web_link=uploaded_file.get("webViewLink"),
        )

        result = await organization_file_collection().insert_one(file_data.model_dump())
        new_file = await organization_file_collection().find_one(
            {"_id": result.inserted_id}
        )

        return OrganizationFileEntity(new_file)

    except HttpError as error:
        print(error)
        raise HTTPException(status_code=500, detail=f"Google Drive API error: {error}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")