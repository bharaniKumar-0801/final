from docx import Document
import io

def extract_text_from_docx(file_content):
    doc = Document(io.BytesIO(file_content))
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

