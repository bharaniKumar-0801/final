from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import sys
from PyPDF2 import PdfReader
from tika import parser as tika_parser

def is_ocr_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    
    # Check if there are any images on the first page
    if '/XObject' in page['/Resources']:
        xObject = page['/Resources']['/XObject'].get_object()
        if xObject:
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    print("Likely an OCR PDF\n")
                    return True  # Likely an OCR PDF
    
    # If we can extract text, it's likely machine-generated
    if page.extract_text().strip():
        return False  # Likely a machine-generated PDF
    
    return True  # If no text extracted, assume it's OCR


def extract_text_from_pdf(pdf_path):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path)
    
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

# Usage
pdf_path = sys.argv[1]
if (is_ocr_pdf(pdf_path)):
    print("\n\n ----------Extracting text from OCR PDF-----------\n\n")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)
else:
    print("\n\n-----Machine generated PDF\n\n")
    parsed_pdf = tika_parser.from_file(pdf_path)
    print(parsed_pdf['metadata'])
    contents = parsed_pdf['content']
    print("\n\n Contents of the file: " + pdf_path)
    print("---------------------\n\n")
    print(contents)
