import pdfplumber
import re
import sys

def extract_core_text(pdf_path):
    core_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract words from the page
            words = page.extract_words(x_tolerance=1, y_tolerance=3)
            
            # Group words into lines
            lines = []
            current_line = []
            last_bottom = -1
            for word in words:
                if word['bottom'] != last_bottom and current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                current_line.append(word['text'])
                last_bottom = word['bottom']
            if current_line:
                lines.append(' '.join(current_line))
            
            # Remove header and footer
            if len(lines) > 2:
                lines = lines[1:-1]  # Remove first and last line
            
            # Process each line
            processed_lines = []
            for line in lines:
                # Remove page numbers
                if not re.match(r'^\d+$', line.strip()):
                    # Remove lines that look like they're from a table of contents
                    #if not re.match(r'^\s*[\w\s]+\s*\.+\s*\d+\s*$', line):
                        # Remove invisible characters and trailing/leading whitespace
                    cleaned_line = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', line).strip()
                    # Only add non-empty lines
                    if len(cleaned_line.strip()) >1:
                        processed_lines.append(cleaned_line)
            
            # Join the remaining lines
            core_text.append('\n'.join(processed_lines))
    
    return '\n'.join(core_text)

# Usage
pdf_path = sys.argv[1]
extracted_text = extract_core_text(pdf_path)
print(extracted_text)
