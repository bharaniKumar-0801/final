from tika import parser
import sys

# Extract text from a PDF file
#text = parser.from_file('example.pdf')['content']
#print(text)


# Extract metadata from a PDF file
filename = sys.argv[1]
print(filename)
parsed_pdf = parser.from_file(filename)
print(parsed_pdf['metadata'])
contents = parsed_pdf['content']
print("\n\n Contents of the file: " + filename)
print("---------------------\n\n")
print(contents)
#print(metadata_dict['metadata']['pdf:docinfo:title'])

