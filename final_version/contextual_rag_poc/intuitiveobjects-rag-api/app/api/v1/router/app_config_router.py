from fastapi import APIRouter, HTTPException, Depends
from app.schema.app_config_schema import (
    CreateAppConfigRequest,
)
import app.services.app_config_service as app_config_service

app_config_router = APIRouter()

@app_config_router.post("/app-config")
async def create_app_config(app_config: CreateAppConfigRequest):
    try:
        print(app_config)
        result = await app_config_service.create_app_config(app_config)
        return {"success": True, "message": "App config created successfully", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_config_router.get("/app-config")
async def get_app_configs():
    try:
        result = await app_config_service.get_app_configs()
        return {"success": True, "message": "App configs retrieved successfully", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
