# import logging
# from .file_processors.file_type_detector import detect_file_type
# from .file_processors.pdf_processor import extract_text_from_pdf
# from .file_processors.docx_processor import extract_text_from_docx
# from .file_processors.image_processor import extract_text_from_image
# from .file_processors.text_processor import extract_text_from_text

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
# )
# logger = logging.getLogger(__name__)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')

# # Add handler (console in this case)
# handler = logging.StreamHandler()
# handler.setFormatter(formatter)
# logger.addHandler(handler)


# def process_document(file_content):
#     file_type = detect_file_type(file_content)
#     logger.info(f"Detected file type: {file_type}")
    
#     if file_type == 'application/pdf':
#         text = extract_text_from_pdf(file_content)
#     elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
#         text = extract_text_from_docx(file_content)
#     elif file_type.startswith('image/'):
#         text = extract_text_from_image(file_content)
#     elif file_type.startswith('text/'):
#         text = extract_text_from_text(file_content)
#     else:
#         logger.error(f"Unsupported file type: {file_type}")
#         return None
    
#     logger.info(f"Extracted text, size: {len(text)} characters")
#     logger.info(text)
#     return text



import logging
import zipfile
from io import BytesIO
from unidecode import unidecode

from .file_processors.file_type_detector import detect_file_type
from .file_processors.pdf_processor import extract_text_from_pdf
from .file_processors.docx_processor import extract_text_from_docx
from .file_processors.image_processor import extract_text_from_image
from .file_processors.text_processor import extract_text_from_text
import threading

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')

# Add handler (console in this case)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def process_document(file_content):
    file_type = detect_file_type(file_content)
    logger.info(f"Detected file type: {file_type}")
    
    
    if file_type == 'application/zip':
        return extract_zip_file_texts(file_content)
    if file_type == 'application/pdf':
        text = extract_text_from_pdf(file_content)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = extract_text_from_docx(file_content)
    elif file_type.startswith('image/'):
        text = extract_text_from_image(file_content)
    elif file_type.startswith('text/'):
        text = extract_text_from_text(file_content)
    else:
        logger.error(f"Unsupported file type: {file_type}")
        return None
    
    logger.info(f"Extracted text, size: {len(text)} characters")
    logger.info(text)
    return text


def extract_zip_file_texts(zip_file_content: bytes) -> dict[str, str]:
    """
    Process a ZIP file, extract and return text for each file inside as {filename: text}
    """
    # extracted_texts = {}

    try:    
        with zipfile.ZipFile(BytesIO(zip_file_content)) as zip_file:
            for zip_entry in zip_file.infolist():
                if zip_entry.is_dir():
                    continue

                entry_name = zip_entry.filename
                logger.info(f"Extracting file from ZIP: {entry_name}")

                with zip_file.open(zip_entry) as file_obj:
                    entry_content = file_obj.read()
                    try:
                        # text = process_document(entry_content)
                    # Start a thread to process the document
                        thread = threading.Thread(
                            target=process_document,
                            args=(entry_content,),
                            daemon=True  # Daemon threads will not block program exit
                        )
                        thread.start()  

                        # if text:
                        #     extracted_texts[entry_name] = text
                    except Exception as e:
                        logger.error(f"Error processing {entry_name}: {e}")
                        # extracted_texts[entry_name] = f"Error: {str(e)}"
    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP file: {e}")
        return  {"error": "Invalid ZIP file"}