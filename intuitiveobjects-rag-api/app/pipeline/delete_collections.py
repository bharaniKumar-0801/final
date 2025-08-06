from chromadb import PersistentClient

import os
PERSIST_DIRECTORY = "/home/vishwa/harry_rag/intuitiveobjects-rag-api/chroma_storage"

os.makedirs(PERSIST_DIRECTORY, exist_ok=True) 

chroma_client = PersistentClient(path=PERSIST_DIRECTORY)

# print("Collections before deletion:", chroma_client.list_collections()) 

# chroma_client.delete_collection("document_collection") 



collection_name = "document_collection"
existing_collections = [col.name for col in chroma_client.list_collections()]

if collection_name in existing_collections:
    chroma_client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted.")
else:
    print(f"Collection '{collection_name}' does not exist.")
