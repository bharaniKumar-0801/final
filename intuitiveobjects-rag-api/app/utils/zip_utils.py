
# import zipfile, io, hashlib
# from gridfs import GridFS
# from bson import ObjectId
# import threading
# from app.pipeline.ingest import process_uploaded_document

# def get_sha256_hash(content: bytes) -> str:
#     return hashlib.sha256(content).hexdigest()

# def process_zip_file(zip_bytes, fs: GridFS, file_collection, metadata_collection):
#     zip_hash = get_sha256_hash(zip_bytes)
#     zip_stream = io.BytesIO(zip_bytes)

#     with zipfile.ZipFile(zip_stream) as zip_file:
#         file_hashes = []
#         for file_name in zip_file.namelist():
#             content = zip_file.read(file_name)
#             file_hash = get_sha256_hash(content)

#             # Store file if not already
#             if not fs.exists({"filename": file_hash}):
#                 fs.put(content, filename=file_hash)

#             # Store metadata
#             metadata_collection.update_one(
#                 {"_id": file_hash},
#                 {"$set": {"filename": file_name, "status": "not_ingested"}},
#                 upsert=True
#             )
#             file_hashes.append(file_hash)

#             # Process each file in a new thread
#             thread = threading.Thread(target=process_uploaded_document, args=(file_hash,))
#             thread.start()

#         # Save zip file itself and its metadata
#         file_collection.update_one(
#             {"_id": zip_hash},
#             {"$set": {"type": "zip", "files": file_hashes}},
#             upsert=True
#         )


import zipfile, io, hashlib
from typing import List
from bson import ObjectId
import threading


from app.pipeline.ingest import process_uploaded_document



def get_sha256_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def process_zip_file_return_contents(zip_bytes: bytes):
    zip_hash = get_sha256_hash(zip_bytes)
    zip_stream = io.BytesIO(zip_bytes)
    file_contents = []

    with zipfile.ZipFile(zip_stream) as zip_file:
        for file_name in zip_file.namelist():
            content = zip_file.read(file_name)
            file_hash = get_sha256_hash(content)

            file_contents.append({
                "_id": file_hash,
                "filename": file_name,
                "content": content.decode('utf-8', errors='ignore'),
                "status": "not_ingested"
            })

            # Background ingestion thread
            thread = threading.Thread(target=process_uploaded_document, args=(file_hash,))
            thread.start()

    return zip_hash, file_contents