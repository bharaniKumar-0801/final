import fitz  # PyMuPDF
import re
import sys
from unidecode import unidecode

def extract_text(pdf_path):
    doc = fitz.open(pdf_path, filetype="pdf")
    for page in doc:
        blocks = page.get_text("blocks")
        previous_block_id = 0 # Set a variable to mark the block id
        for block in blocks:
            if block[6] == 0: # We only take the text
                #if previous_block_id != block[5]: # Compare the block number
                    #print("\n")
                plain_text = unidecode(block[4])
            print(plain_text)

def extract_core_text(pdf_path):
    core_text = []
    
    doc = fitz.open(pdf_path, filetype="pdf")
    
    for page in doc:
        text = page.get_text()
        
        # Split text into lines
        lines = text.split('\n')
        
        # Process each line
        processed_lines = []
        for line in lines:
            # Remove invisible characters
            cleaned_line = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', line)
            # Remove extra spaces
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
            
            # Only add non-empty lines with more than one character
            if len(cleaned_line) > 1:
                processed_lines.append(cleaned_line)
        
        # Join the remaining lines
        core_text.append('\n'.join(processed_lines))
    
    doc.close()
    return '\n'.join(core_text)

if __name__ == "__main__":
    # Usage
    pdf_path = sys.argv[1]
    #extracted_text = extract_core_text(pdf_path)
    extracted_text = extract_text(pdf_path)
    print(extracted_text)
