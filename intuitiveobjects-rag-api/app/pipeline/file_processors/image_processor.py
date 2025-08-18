from PIL import Image
import pytesseract
import io

def extract_text_from_image(file_content):
    image = Image.open(io.BytesIO(file_content))
    return pytesseract.image_to_string(image)
