from pymongo import MongoClient
from gridfs import GridFS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_mongodb():
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/rag_ui")
        db = client["database_name"]
        
        # Remove all files from GridFS
        fs = GridFS(db)
        fs_files = db["fs.files"]
        fs_chunks = db["fs.chunks"]
        
        logger.info("Removing all files from GridFS...")
        fs_files.delete_many({})
        fs_chunks.delete_many({})
        logger.info("All files removed from GridFS.")
        
        # List all collections in the database
        collections = db.list_collection_names()
        
        # Remove all documents from each collection (except system collections)
        for collection_name in collections:
            if not collection_name.startswith("system."):
                collection = db[collection_name]
                result = collection.delete_many({})
                logger.info(f"Removed {result.deleted_count} documents from {collection_name}")
        
        logger.info("MongoDB cleanup completed successfully.")
        return True
    
    except Exception as e:
        logger.error(f"An error occurred while cleaning up MongoDB: {str(e)}")
        return False

if __name__ == "__main__":
    success = cleanup_mongodb()
    if success:
        logger.info("MongoDB cleanup completed successfully.")
    else:
        logger.error("Failed to clean up MongoDB.")
