import pdfplumber
import re
import sys

def extract_core_text(pdf_path):
    core_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text from the page
            #text = page.extract_text()
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            
            # Split text into lines
            lines = text.split('\n')
            
            # Remove header and footer
            if len(lines) > 2:
                lines = lines[1:-1]  # Remove first and last line
            
            # Process each line
            processed_lines = []
            for line in lines:
                # Remove page numbers
                if not re.match(r'^\d+$', line.strip()):
                    # Remove lines that look like they're from a table of contents
                    if not re.match(r'^\s*[\w\s]+\s*\.+\s*\d+\s*$', line):
                        # Remove lines with only one visible character
                        if len(line.strip()) > 1:
                            processed_lines.append(line)

            # Join the remaining lines
            core_text.append('\n'.join(processed_lines))
            
    return '\n'.join(core_text)

# Usage
pdf_path = sys.argv[1]
extracted_text = extract_core_text(pdf_path)
print(extracted_text)
