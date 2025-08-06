



import os
import io
import zipfile
import hashlib
import threading
import logging
import copy

from bson import ObjectId
from pymongo import MongoClient
from gridfs import GridFS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# App Modules
from app.pipeline.document_processor import process_document
from app.pipeline.chroma_db import chroma_add_chunks
from app.pipeline.file_processors.extaract_metadata import generate_document_metadata
from app.pipeline.models import get_model_manager
from app.pipeline.keyword_search.ingest_text import save_bm25_index
from app.pipeline.progress_tracker import set_progress
from app.pipeline.logger_config import setup_logger
from app.utils.metadata_utils import normalize_metadata

# ────────────────────────────────────────────────────────────────────────────────
# 🔧 Setup & Initialization
# ────────────────────────────────────────────────────────────────────────────────

logger = setup_logger(__name__, log_level=logging.DEBUG)

mongo_client = MongoClient("mongodb://localhost:27017/rag_ui")
db = mongo_client["database_name"]
fs = GridFS(db)
fs_files = db.fs.files

model_manager = get_model_manager()


def get_gridfs() -> GridFS:
    """Thread-safe access to GridFS."""
    return fs


# ────────────────────────────────────────────────────────────────────────────────
# 📄 Chunking Utility
# ────────────────────────────────────────────────────────────────────────────────

def chunk_document(pages: list[str], metadata: dict, chunk_size=500, overlap=100) -> list[dict]:
    """
    Chunk document into sections with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", "", "."]
    )
    full_text = "\n".join(pages).strip()
    chunks = text_splitter.split_text(full_text)

    logger.info(f"Chunked into {len(chunks)} sections.")


    normalized_metadata = normalize_metadata(metadata)

    return [
        {
            "chunk_id": f"chunk_{i}",
            "text": chunk,
            "metadata": copy.deepcopy(normalized_metadata)  # Avoid shared reference
        }
        for i, chunk in enumerate(chunks)
    ]



# ────────────────────────────────────────────────────────────────────────────────
# 🔁 Main Processing Function
# ────────────────────────────────────────────────────────────────────────────────

def process_uploaded_document(file_id_str: str):
    """
    Main pipeline:
    1. Fetch from GridFS
    2. Extract text
    3. Generate metadata
    4. Chunk + BM25
    5. Store in ChromaDB
    6. Mark complete
    """
    try:
        file_id = file_id_str
        logger.info(f"🚀 Processing document ID: {file_id}")

        fs = get_gridfs()

        if not fs.exists(file_id):
            logger.error(f"❌ File ID {file_id} not found.")
            return

        file_content = fs.get(file_id).read()
        logger.info(f"📥 Loaded {len(file_content)} bytes from GridFS.")

        # Step 1: Extract text
        pages = process_document(file_content)
        if not pages:
            logger.warning("⚠️ No text extracted from file.")
            return

        # Step 2: Keyword Search Index (BM25)
        


        # Step 3: Generate metadata via LLM
        model_name = model_manager.initialize_gemma_model()
        metadata = generate_document_metadata(pages, model_name)  

        normalized_metadata = normalize_metadata(metadata)

        logger.info(f"Generated metadata: {normalized_metadata}")

        save_bm25_index(file_id, pages)

        logger.info("✅ BM25 storage completed.")


        # Step 4: Chunk for Chroma
        chunks = chunk_document(pages, normalized_metadata) 

        logger.info(f"Chunked into {chunks} chunks.")

        # Step 5: Embed and store in Chroma
        chroma_add_chunks(chunks, file_id) 

        logger.info("✅ ChromaDB storage completed.")

        # save_bm25_index(file_id, chunks) 

        # logger.info("✅ BM25 storage completed.")

        # Step 6: Update file status
        fs_files.update_one(
            {"_id": file_id},
            {"$set": {"metadata.status": "completed"}}
        )

    except Exception as e:
        logger.error(f"❌ Error processing document ID {file_id_str}: {e}", exc_info=True)
        set_progress(file_id_str, -1)
        fs_files.update_one(
            {"_id": file_id_str},
            {"$set": {"metadata.status": "failed"}}
        )


# ────────────────────────────────────────────────────────────────────────────────
# 🗂️ ZIP Extraction Handler
# ────────────────────────────────────────────────────────────────────────────────

def extract_zip_and_store_files(file_id: str):
    """
    1. Extract files from ZIP
    2. Deduplicate by hash
    3. Store in GridFS
    4. Process each file in background thread
    """
    def worker(zip_info, inner_content: bytes):
        try:
            inner_hash = hashlib.sha256(inner_content).hexdigest()

            if not fs.exists({"_id": inner_hash}):
                fs.put(
                    io.BytesIO(inner_content),
                    _id=inner_hash,
                    filename=zip_info.filename,
                    metadata={
                        "source_zip_id": file_id,
                        "original_filename": zip_info.filename,
                        "contentType": "application/octet-stream",
                        "status": "pending"
                    }
                )
                logger.info(f"🧾 Stored file: {zip_info.filename} (ID: {inner_hash})")
            else:
                logger.info(f"🟡 Duplicate skipped: {zip_info.filename}")

            # Begin processing
            process_uploaded_document(inner_hash)

        except Exception as e:
            logger.error(f"❌ Error in file {zip_info.filename}: {e}", exc_info=True)

    try:
        logger.info(f"📦 Extracting ZIP file ID: {file_id}")
        zip_bytes = fs.get(file_id).read()
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))

        for zip_info in zip_file.infolist():
            if zip_info.is_dir():
                continue

            with zip_file.open(zip_info) as inner_file:
                inner_content = inner_file.read()
                threading.Thread(
                    target=worker,
                    args=(zip_info, inner_content),
                    daemon=True
                ).start()

        logger.info("✅ All ZIP files dispatched for processing.")

    except Exception as e:
        logger.error(f"❌ ZIP extraction failed for ID {file_id}: {e}", exc_info=True)
