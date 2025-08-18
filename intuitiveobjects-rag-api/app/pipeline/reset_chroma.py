


import chromadb
from chromadb.utils import embedding_functions
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

def reset_chroma_collection():
    try:
        PERSIST_DIRECTORY = "/home/bharani/final_integration/intuitiveobjects-rag-api/chroma_storage"
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        if "document_collection" in [c.name for c in chroma_client.list_collections()]:
            logger.info("Existing collection found. Deleting it...")
            chroma_client.delete_collection("document_collection")
            logger.info("Collection deleted.")

        # Optionally re-init client to avoid cache issues
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

        new_collection = chroma_client.create_collection(
            name="document_collection",
            embedding_function=embedding_function
        )
        logger.info("New collection created successfully.")

        return True

    except Exception as e:
        logger.error(f"An error occurred while resetting the ChromaDB collection: {str(e)}")
        return False

if __name__ == "__main__":
    if reset_chroma_collection():
        logger.info("ChromaDB collection reset completed successfully.")
    else:
        logger.error("Failed to reset ChromaDB collection.")
