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

