from chromadb.config import Settings
import chromadb

client = chromadb.Client(Settings(persist_directory="./chroma_storage"))

collections = client.list_collections()

for col in collections:
    print(f"Collection name: {col.name}")
