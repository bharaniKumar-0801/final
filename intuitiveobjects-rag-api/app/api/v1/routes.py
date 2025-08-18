from fastapi import APIRouter, UploadFile, File
from app.pipeline.ingest import process_uploaded_document
from app.api.v1.router.user_router import user_router
from app.api.v1.router.chat_router import chat_router
from app.api.v1.router.organization_router import organization_router
from app.api.v1.router.organization_admin_router import organization_admin_router
from app.api.v1.router.organization_user_router import organization_user_router
from app.api.v1.router.organization_file_router import organization_file_router
from app.api.v1.router.app_config_router import app_config_router

from app.api.v1.router.reset_db_router import reset_chroma_router
from app.api.v1.router.reset_db_router import reset_mongo_router


router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save file to disk or memory as needed
    # Call your pipeline function
    process_uploaded_document(file)
    return {"status": "processing started"}

router.include_router(user_router, prefix="/user")
router.include_router(chat_router, prefix="/chat")
router.include_router(organization_router, prefix="/organization")
router.include_router(organization_admin_router, prefix="/organization-admin")
router.include_router(organization_user_router, prefix="/organization-user")
router.include_router(organization_file_router, prefix="/organization-file")
router.include_router(app_config_router, prefix="/organization-app-config")


router.include_router(reset_chroma_router, prefix="/admin/reset-chroma")
router.include_router(reset_mongo_router, prefix="/admin/reset-mongo")
