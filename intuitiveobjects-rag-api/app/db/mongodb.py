from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo.errors import ConnectionFailure
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class Database:
    client: AsyncIOMotorClient = None
    fs: AsyncIOMotorGridFSBucket = None
    


async def connect_to_mongodb():
    """Connect to MongoDB Atlas."""
    Database.client = AsyncIOMotorClient(
        settings.MONGODB_URI, serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT
    )
    try:
        await Database.client.admin.command("ping")
        logger.info("Connected to MongoDB")
        print("Connected to MongoDB")
        Database.fs = AsyncIOMotorGridFSBucket(Database.client[settings.MONGODB_DB])
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")

        raise e


async def close_mongodb_connection():
    """Close the MongoDB connection."""
    if Database.client:
        Database.client.close()
        logger.info("Closed connection with MongoDB")


def get_database():
    """Return the database instance."""
    return Database.client[settings.MONGODB_DB]


def get_collection(name: str):
    """Get a specific collection."""
    return get_database()[name]


def user_collection():
    """Get the user collection."""
    return get_collection("users")


def token_collection():
    """Get the token collection."""
    return get_collection("tokens")


def chat_collection():
    """Get the chat collection."""
    return get_collection("chats")


def message_collection():
    """Get the message collection."""
    return get_collection("messages")


def organization_user_collection():
    """Get the organization user collection."""
    return get_collection("organization_users")


def organization_admin_collection():
    """Get the organization admin collection."""
    return get_collection("organization_admins")


def organization_file_collection():
    """Get the organization file collection."""
    return get_collection("organization_files")



def organization_collection():
    """Get the organization collection."""
    return get_collection("organizations")


def category_collection():
    """Get the category collection."""
    return get_collection("categories")


def organization_app_config_collection():
    """Get the organization app config collection."""
    return get_collection("organization_app_configs")


def app_config_collection():
    """Get the app config collection."""
    # return get_collection("app_configs")
    return get_collection("organization_app_configs")

def get_fs():
    """Get the GridFS bucket."""
    return Database.fs
    # Database.fs = AsyncIOMotorGridFSBucket(
    #     settings.MONGODB_URI, serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT
    # )
    # return Database.fs
