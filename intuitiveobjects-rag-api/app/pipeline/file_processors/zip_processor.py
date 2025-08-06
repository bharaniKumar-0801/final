import io
import zipfile
import logging

from .file_type_detector import detect_file_type
from .pdf_processor import extract_text_from_pdf
from .docx_processor import extract_text_from_docx
from .image_processor import extract_text_from_image
from .text_processor import extract_text_from_text

logger = logging.getLogger(__name__)


def process_zip_file(zip_bytes):
    extracted_texts = {}

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_ref:
            for name in zip_ref.namelist():
                logger.info(f"Extracting and processing file: {name}")
                with zip_ref.open(name) as f:
                    file_data = f.read()
                    file_type = detect_file_type(file_data)

                    if file_type == "application/pdf":
                        text = extract_text_from_pdf(file_data)
                    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = extract_text_from_docx(file_data)
                    elif file_type.startswith("image/"):
                        text = extract_text_from_image(file_data)
                    elif file_type.startswith("text/"):
                        text = extract_text_from_text(file_data)
                    else:
                        logger.warning(f"Unsupported file type inside zip: {file_type}")
                        text = None

                    extracted_texts[name] = text or ""

    except Exception as e:
        logger.error(f"Error while processing ZIP file: {e}", exc_info=True)
        return None

    return extracted_texts
