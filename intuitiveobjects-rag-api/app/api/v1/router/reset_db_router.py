from fastapi import APIRouter
from app.pipeline.reset_chroma import reset_chroma_collection
from app.pipeline.cleanup_mongodb import cleanup_mongodb

reset_chroma_router = APIRouter()

@reset_chroma_router.post("/")
def reset_chroma():
    success =  reset_chroma_collection()
    return {"success": success}


reset_mongo_router = APIRouter()

@reset_mongo_router.post("/")
def reset_mongo():
    success = cleanup_mongodb()
    return {"success": success}
