import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # API settings
    PROJECT_NAME: str = "Demo API"
    API_V1_STR: str = "/api/v1"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "secret_key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_DB: str = os.getenv("MONGODB_DB")
    MONGODB_USER: str = os.getenv("MONGODB_USER")
    MONGODB_PASSWORD: str = os.getenv("MONGODB_PASSWORD")
    MONGODB_TIMEOUT: int = 3000
    
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER")
    GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE: str = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()