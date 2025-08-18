def AppConfigEntity(app_config) -> dict:
    return {
        "id": str(app_config["_id"]),
        "llm_model": app_config.get("llm_model"),
        "embedding_model": app_config.get("embedding_model"),
        "temperature": app_config.get("temperature", 0.7),  # returns None if missing
        "created_at": app_config.get("created_at"),
        "updated_at": app_config.get("updated_at"),
    }

def AppConfigListEntity(app_configs) -> list:
    return [AppConfigEntity(app_config) for app_config in app_configs]
