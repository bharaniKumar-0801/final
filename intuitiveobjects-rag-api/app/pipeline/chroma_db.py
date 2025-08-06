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

# Set up logging
import logging
from .logger_config import setup_logger
logger = setup_logger(__name__, log_level=logging.DEBUG)

# Set up Groq client (not used in this example, but kept for future use)
#groq_api_key = os.getenv("GROQ_API_KEY")
#groq_client = Client(api_key=groq_api_key)

# Set up ChromaDB client
# PERSIST_DIRECTORY = os.path.join(os.getcwd(), "/home/vishwa/harry_rag/intuitiveobjects-rag-api/chroma_storage")
# os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
PERSIST_DIRECTORY = "/home/bharani/final_integration/intuitiveobjects-rag-api/chroma_storage"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

#chroma_client = chromadb.Client()

# # After adding documents, you can verify the collection exists
# collection_names = chroma_client.list_collections()
# logger.info(f"Available collections: {collection_names}")

# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# # collection = chroma_client.get_or_create_collection(
# #     name="document_collection",
# #     embedding_function=embedding_function
# # )

# # Print the embedding dimension

# test_embedding = embedding_function(["test"])
# embedding_dim = len(test_embedding[0])
# logger.info(f"Embedding dimension in chroma_db: {embedding_dim}")

# def get_embeddings(chunks):
#     embeddings = embedding_function(chunks)
#     logger.info(f"Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0])}")
#     return embeddings


from app.services.app_config_service import get_app_configs 
import app.services.organization_file_services as organization_file_services

async def init_chroma_collection():

    global collection, embedding_function

    app_config= await get_app_configs()
    model_name = app_config['embedding_model']  # Get the embedding model from app config
    logger.info(f"Using embedding model: {model_name}") 

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = chroma_client.get_or_create_collection(
        name="document_collection",
        embedding_function=embedding_function
    )

    test_embedding = embedding_function(["test"])
    # embedding_dim = 
    logger.info(f"Embedding dimension in chroma_db: {len(test_embedding[0])}")

# def get_embeddings(chunks):
#     if embedding_function is None:
#         raise ValueError("Embedding function is not initialized. Please call init_chroma_collection first.")
    
#     embeddings = embedding_function(chunks)
#     logger.info(f"Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0])}")
#     return embeddings   
 



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
        logger.exception(f"❌ Failed to store chunks in ChromaDB for file_id={file_id}: {e}")



# def chroma_add_chunks(chunks, file_id):


#     collection = chroma_client.get_or_create_collection(
#     name="document_collection",
#     embedding_function=embedding_function
# )
#     # Generate embeddings from chunk texts
#     embeddings = get_embeddings([chunk["text"] for chunk in chunks])
    
#     # Use the file_id as a prefix for chunk IDs
#     ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
    
#     # Prepare detailed metadata per chunk, merging your chunk metadata dict
#     chunk_metadatas = [
#         {
#             "source": str(file_id),
#             "chunk_index": i,
#             "total_chunks": len(chunks),
#             **chunk.get("metadata", {}),  # include chunk metadata if exists
#             # "page": chunk.get("page")    # include page number if exists
#         }
#         for i, chunk in enumerate(chunks)
#     ]
#     logger.info(f"Chunk metadata prepared for {len(chunk_metadatas)} chunks")
#     try:
#         collection.add(
#             ids=ids,
#             embeddings=embeddings,
#             documents=[chunk["text"] for chunk in chunks],  # must be list of strings
#             metadatas=chunk_metadatas
#         )
#         logger.info(f"Number of embeddings: {len(embeddings)}")
#         logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
#         collection_names = chroma_client.list_collections()
#         logger.info(f"Available collections: {collection_names}")
#         for c in collection_names:
#             print(c.name)
#     except Exception as e:
#         logger.error(f"Error storing chunks in ChromaDB: {str(e)}")




'''
def chroma_add_documents(documents, metadatas):
    embeddings = get_embeddings(documents)
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    logger.info(f"Added {len(documents)} documents to ChromaDB")

# Set up MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["rag_documents"]
fs = GridFS(db)

def chunk_document(document, chunk_size=1000):
    #"""
    #Chunk a document into smaller pieces.
    
    #Args:
    #    document (str): The document to be chunked.
    #    chunk_size (int): The maximum length of each chunk.
    
    #Returns:
    #    List[str]: A list of chunked documents.
    #"""

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", "", "."]
    )
    chunks = text_splitter.split_text(document)
    logger.info(f"Document chunked into {len(chunks)} pieces")
    for i, chunk in enumerate(chunks):
        logger.info(f"-----Chunk num {i} ----")
        logger.info({chunk})
        logger.info("\n\n")
    return chunks


def get_gridfs():
    #Create a new GridFS instance for thread-safe access.
    return GridFS(db)

def old_get_embeddings(chunks):
    #"""
    #Generate embeddings for a list of text chunks using Ollama.
    
    #Args:
    #    chunks (List[str]): A list of text chunks.
    
    #Returns:
        List[List[float]]: A list of embeddings.
    #"""
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )
            embedding = response["embedding"]
            embeddings.append(embedding)
            logger.debug(f"Generated embedding for chunk {i+1}/{len(chunks)}")
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {i+1}: {str(e)}")
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings

#Moved process_uploaded_document to ingest.py

def process_uploaded_document(file_id_str):
    #"""
    #Process an uploaded document: retrieve from MongoDB, chunk it,
    #generate embeddings, and store in ChromaDB.
    
    #Args:
        file_id_str: String form of the ID of the file in MongoDB GridFS.
    #"""
    # Convert string ID to ObjectId
    file_id = ObjectId(file_id_str)
    logger.info(f"Starting to process document with ID {file_id}")
    
    try:
        
        
        # Get a thread-local GridFS instance
        thread_fs = get_gridfs()

        # Retrieve the document from MongoDB
        if not thread_fs.exists(file_id):
            logger.error(f"File with ID {file_id} does not exist in GridFS")
            return

        grid_file = thread_fs.get(file_id)
        file_content = grid_file.read()
        
        text = process_document(file_content)
        if text is None:
            return
        
        logger.info(f"Retrieved document from MongoDB, size: {len(file_content)} characters")
        
        # Chunk the document
        chunks = chunk_document(text)
        logger.info(f"Document chunked into {len(chunks)} pieces")
        
        # Generate embeddings
        embeddings = get_embeddings(chunks)
        
        # Store in ChromaDB
        # Use the file_id as a prefix for chunk IDs
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        
        # More detailed metadata example:
        chunk_metadatas = [
            {
                "source": str(file_id),
                "chunk_index": i,
                "total_chunks": len(chunks)
            } 
            for i, _ in enumerate(chunks)
        ]
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                #metadatas=[{"source": str(file_id)} for _ in chunks]
                metadatas=chunk_metadatas
            )
            logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
        except Exception as e:
            logger.error(f"Error storing chunks in ChromaDB: {str(e)}")
        
        logger.info(f"Completed processing document with ID {file_id}")
        logger.info(f"Number of chunks: {len(chunks)}")
        logger.info(f"Number of embeddings: {len(embeddings)}")
    except Exception as e:
        logger.error(f"Error processing document with ID {file_id}: {str(e)}")
        logger.exception("Full traceback:")

'''

