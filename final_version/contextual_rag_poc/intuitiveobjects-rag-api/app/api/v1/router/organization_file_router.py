

from fastapi import APIRouter, UploadFile, File, Form, Depends
from app.utils.auth import get_current_user
# from app.pipeline.ingest import extract_zip_and_store_files
from app.pipeline.ingest import process_uploaded_document
from fastapi import HTTPException

from app.pipeline.ingest import extract_zip_and_store_files
import app.services.organization_file_services as organization_file_services
from pymongo import MongoClient
from gridfs import GridFS
from typing import List
import io
import json
import logging
import threading
import hashlib
from app.pipeline.progress_tracker import get_progress  # ✅ not progress_map
# from app.pipeline.progress_tracker import process_uploaded_document

from fastapi import UploadFile
import zipfile
from io import BytesIO

# from app.utils.hash import get_file_hash


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# MongoDB and GridFS setup
mongo_client = MongoClient("mongodb://mongodb:27017/rag_ui")  # adjust host as needed
db = mongo_client["database_name"]
fs = GridFS(db)

organization_file_router = APIRouter()




@organization_file_router.post("/upload")
async def upload_file(
    category_id: str = Form(...),
    file: UploadFile = File(...),   
    tags: str = Form(...),
    user_id: str = Depends(get_current_user),
):
    try:
        tag_list = json.loads(tags)
        logger.info(f"Received file upload request from user {user_id}")

        # 🔄 Read file content and compute hash
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        file_stream = io.BytesIO(content)

        # 🔍 Check if file already exists using synchronous fs
        if not fs.exists({"_id": file_hash}):
            file_id = fs.put(
                file_stream,
                _id=file_hash,
                filename=file.filename,
                metadata={
                    "category_id": category_id,
                    "user_id": user_id,
                    "tags": tag_list,
                    "contentType": file.content_type,
                    "status": "progress",
                }
            )
            logger.info(f"File uploaded to GridFS with ID: {file_id}")
        else:
            logger.info(f"Duplicate file detected. Skipping upload for ID: {file_hash}")

        file_id = file_hash  # Consistent with your naming

        # 🧾 Optionally store metadata in collection
        result = await organization_file_services.organization_upload_file(
            category_id, file, tag_list, user_id ,file_id
        )

        if file.content_type == "application/x-zip-compressed":
            extract_zip_and_store_files(file_id)
        else:
            process_uploaded_document(file_id)


        return {"message": "File uploaded successfully", "file_id": str(file_id)}

    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}", exc_info=True)
        return {"error": str(e)}



@organization_file_router.get("/all")
async def get_files(user_id: str = Depends(get_current_user)):
    result = await organization_file_services.organization_get_files(user_id)
    return {"message": "Files fetched successfully", "success": True, "data": result}


@organization_file_router.delete("/{file_id}")
async def delete_file(file_id: str, user_id: str = Depends(get_current_user)):
    result = await organization_file_services.organization_delete_file(file_id, user_id)
    return {"message": "File deleted successfully", "success": True, "data": result}


@organization_file_router.post("/google-drive/upload")
async def upload_google_drive_file(
    category_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    print(category_id, file, user_id)
    result = await organization_file_services.organization_google_drive_upload_file(
        category_id, file, user_id
    )
    return {"message": "File uploaded successfully", "data": result}



@organization_file_router.get("/upload-status/{file_id}")
async def check_upload_status(file_id: str):
    fs_files = db.fs.files
    try:
        file_doc = fs_files.find_one({"_id": file_id})
        if not file_doc or "metadata" not in file_doc:
            return {
                "progress": 0,
                "completed": 0,
                "failed": 0,
                "total": 0,
                "status": "waiting",
                "filename": None,
                "type": "unknown"
            }
        metadata = file_doc["metadata"]
        # Case 1: ZIP-extracted file (has source_zip_id)
        if "source_zip_id" in metadata:
            source_zip_id = metadata["source_zip_id"]
            # Find all files extracted from this ZIP
            extracted_files = list(fs_files.find({"metadata.source_zip_id": source_zip_id}))
            total_files = len(extracted_files)
            completed_files = sum(1 for f in extracted_files if f["metadata"].get("status") == "completed")
            failed_files = sum(1 for f in extracted_files if f["metadata"].get("status") == "failed")
            progress_percent = int((completed_files / total_files) * 100) if total_files > 0 else 0
            status = metadata.get("status", "pending")
            return {
                "progress": progress_percent,
                "completed": completed_files,
                "failed": failed_files,
                "total": total_files,
                "status": status,
                "filename": file_doc.get("filename"),
                "type": "zip_child"
            }
        # Case 2: ZIP parent file (contentType is zip)
        if metadata.get("contentType") == "application/x-zip-compressed":
            extracted_files = list(fs_files.find({"metadata.source_zip_id": file_id}))
            total_files = len(extracted_files)
            completed_files = sum(1 for f in extracted_files if f["metadata"].get("status") == "completed")
            failed_files = sum(1 for f in extracted_files if f["metadata"].get("status") == "failed")
            if total_files == 0:
                return {
                    "progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "total": 0,
                    "status": "waiting",
                    "filename": file_doc.get("filename"),
                    "type": "zip"
                }
            progress_percent = int((completed_files / total_files) * 100)
            status = "done" if (completed_files + failed_files) >= total_files else "in_progress"
            return {
                "progress": progress_percent,
                "completed": completed_files,
                "failed": failed_files,
                "total": total_files,
                "status": status,
                "filename": file_doc.get("filename"),
                "type": "zip"
            }
        # Case 3: Single file (not from ZIP, not a ZIP itself)
        status = metadata.get("status", "pending")
        progress = 100 if status == "completed" else 0
        completed = 1 if status == "completed" else 0
        failed = 1 if status == "failed" else 0
        total = 1
        return {
            "progress": progress,
            "completed": completed,
            "failed": failed,
            "total": total,
            "status": status,
            "filename": file_doc.get("filename"),
            "type": "single"
        }
    except Exception as e:
        logger.error(f"Error retrieving upload progress: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Progress tracking failed")




















