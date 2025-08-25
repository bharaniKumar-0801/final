import chromadb
import ollama
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
#from groq import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient  # Ensure you import MongoClient
import pymongo
from gridfs import GridFS

from bson import ObjectId
from .document_processor import process_document


import logging
from .logger_config import setup_logger
logger = setup_logger(__name__, log_level=logging.DEBUG)


PERSIST_DIRECTORY = "chroma_storage"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)




from app.services.app_config_service import get_app_configs 
import app.services.organization_file_services as organization_file_services

async def init_chroma_collection():

    global collection, embedding_function

    # app_config= await get_app_configs()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Get the embedding model from app config
    logger.info(f"Using embedding model: {model_name}")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = chroma_client.get_or_create_collection(
        name="document_collection",
        embedding_function=embedding_function
    )

    test_embedding = embedding_function(["test"])
    # embedding_dim = 
    logger.info(f"Embedding dimension in chroma_db: {len(test_embedding[0])}")


 



def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using your embedding function."""
    if not texts:
        return []
    return embedding_function(texts)  # Ensure this returns list of list[float]

def chroma_add_chunks(chunks: list[dict], file_id: str) -> None:
    """Add document chunks and their metadata to ChromaDB."""
    
    if not chunks:
        logger.warning(f"No chunks to store for file_id: {file_id}")
        return
    
    try:
        collection = chroma_client.get_or_create_collection(
            name="document_collection",
            embedding_function=embedding_function
        )

        texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
        if not texts:
            logger.warning(f"All chunks are empty for file_id: {file_id}")
            return

        embeddings = get_embeddings(texts)
        if not embeddings:
            logger.error(f"Failed to generate embeddings for file_id: {file_id}")
            return

        ids = [f"{file_id}_chunk_{i}" for i in range(len(texts))]

        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": str(file_id),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if "metadata" in chunk:
                metadata.update(chunk["metadata"])
            chunk_metadatas.append(metadata)
        logger.info(f"Chunk metadata prepared for {chunk_metadatas} chunks") 



        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=chunk_metadatas
        )

        logger.info(f"✅ Stored {len(texts)} chunks for file_id: {file_id}")
        logger.debug(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")

    except Exception as e:
        logger.exception(f"Failed to store chunks in ChromaDB for file_id={file_id}: {e}")



