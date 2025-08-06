from PyPDF2 import PdfReader
import io
import re
from typing import List, Tuple
import logging
import json

import fitz  # PyMuPDF
import sys
from unidecode import unidecode

from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image 
# from app.pipeline.models import get_model_manager, llm_generate_response 

# from app.pipeline.file_processors.extaract_metadata import generate_document_metadata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_using_pdfreader(file_content):
    pdf = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text




def extract_page_texts(file_content: bytes) -> list[str]:
    """
    Extracts and returns a list of cleaned text for each page in the PDF.
    """
    doc = fitz.open(stream=file_content, filetype="pdf")
    logger.info(f"Opened PDF document with {len(doc)} pages")
    
    page_texts = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        text_chunks = [unidecode(block[4]) for block in blocks if block[6] == 0]
        page_text = "".join(text_chunks).strip()
        page_texts.append(page_text)
        logger.debug(f"Extracted text from page {page_num}, length = {len(page_text)}")
    
    return page_texts  




# model_manager = get_model_manager()


# def set_active_model(model_name):
#     try:
#         model_manager.set_active_model(model_name)
#     except ValueError as e:
#         logger.error(f"Failed to set active model: {str(e)}")
#         raise

# def initialize_models():
#     try:
#         model_manager.init_models()
#         logger.info(f"Active model set to: {model_manager.active_model}")
#     except ValueError as e:
#         logger.error(f"Failed to initialize models: {str(e)}")
#         raise


# def get_metadata_from_llm(text: str, page_num: int) -> dict:
#     """
#     Calls an LLM to extract metadata from a page's text.
#     Ensures return value is always a dictionary.
#     """
#     prompt = f"""
# You are a metadata extractor. Extract structured metadata from the following page content.

# Page {page_num} content:
# {text}
    
# Return metadata in JSON format.
#     """

#     model_name = model_manager.initialize_qwen_model()
#     response = llm_generate_response(model_name, prompt, max_length=200)

#     # Try to parse response as JSON
#     try:
#         metadata = json.loads(response)
#         if isinstance(metadata, dict):
#             return metadata
#         else:
#             return {"raw_metadata": response}
#     except Exception:
#         return {"raw_metadata": response}




# def fitz_extract_text(file_content: bytes) -> str:
#     doc = fitz.open(stream=file_content, filetype="pdf")
#     logger.info(f"Opened PDF document with {len(doc)} pages")
#     plain_text = ""
#     for page in doc:
#         blocks = page.get_text("blocks")
#         previous_block_id = 0 # Set a variable to mark the block id
#         for block in blocks:
#             if block[6] == 0: # We only take the text
#                 #if previous_block_id != block[5]: # Compare the block number
#                     #print("\n")
#                 plain_text += unidecode(block[4])
#     logger.info(f"Length of text content in PDF file =  {len(plain_text)}")
#     return plain_text


def fitz_extract_text(file_content: bytes) -> list[str]:
    doc = fitz.open(stream=file_content, filetype="pdf")
    logger.info(f"Opened PDF document with {len(doc)} pages")
    page_texts = []
    for page in doc:
        blocks = page.get_text("blocks")
        page_text = ""
        for block in blocks:
            if block[6] == 0:
                page_text += unidecode(block[4])
        page_texts.append(page_text.strip())
    return page_texts



def ocr_extract_text(byte_stream):
    # Convert PDF to list of images
    images = convert_from_bytes(byte_stream)
    
    # Extract text from each image
    text = ""
    for image in images:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Use pytesseract to do OCR on the image
        text += pytesseract.image_to_string(Image.open(io.BytesIO(img_byte_arr)))
    
    return text

def extract_text_from_pdf(file_content: bytes) -> list[dict]:
    # plain_text = fitz_extract_text(file_content)
    page_texts = fitz_extract_text(file_content)
    if page_texts == "":
        logger.info(f"PyMuPDF failed to extract....treating this as OCR content")
        page_texts = ocr_extract_text(file_content)
    return page_texts
        

def clean_text(text: str) -> str:
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove common watermark texts (customize as needed)
    watermark_patterns = [
        r'CONFIDENTIAL',
        r'DRAFT',
        r'DO NOT COPY',
        # Add more patterns as needed
    ]
    for pattern in watermark_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove page numbers
    text = re.sub(r'\b\d+\b(?:\s*of\s*\d+)?', '', text)

    return text.strip()

