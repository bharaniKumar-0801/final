import PyPDF2
import re
import sys

def extract_core_text(pdf_path):
    core_text = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page in reader.pages:
            text = page.extract_text()
            
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
    
    return '\n'.join(core_text)

# Usage
pdf_path = sys.argv[1]
extracted_text = extract_core_text(pdf_path)
print(extracted_text)
