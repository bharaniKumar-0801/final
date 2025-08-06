from app.db.mongodb import app_config_collection
from app.schema.app_config_schema import CreateAppConfigRequest
from app.models.app_config_model import AppConfig
from app.serializers.app_config_serializers import AppConfigEntity, AppConfigListEntity
from datetime import datetime

async def create_app_config(app_config: CreateAppConfigRequest):
    app_config = AppConfig(
        llm_model=app_config.llm_model,
        embedding_model=app_config.embedding_model,
    )
    result = await app_config_collection().insert_one(app_config.model_dump())
    app_config = await app_config_collection().find_one({"_id": result.inserted_id})
    return AppConfigEntity(app_config)


# async def get_app_configs():
#     app_configs = await app_config_collection().find().to_list()
#     return AppConfigListEntity(app_configs)

async def get_app_configs():
    app_config = await app_config_collection().find_one(sort=[("updated_at", -1)])
    if not app_config:
        # raise ValueError("No app configurations found")
        now = datetime.utcnow()
        default_config = {
            "llm_model": "gemma",
            "embedding_model": "all-MiniLM-L6-v2",
            "temperature": 0.7,
            "system_prompt": "hi you are llm model for answering the question ",
            "organization_id": "default_org",
            "created_at": now,
            "updated_at": now,
            
        }
        await app_config_collection().insert_one(default_config)
        app_config = await app_config_collection().find_one(sort=[("updated_at", -1)])
    return AppConfigEntity(app_config)