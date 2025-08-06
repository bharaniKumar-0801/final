from tabula import read_pdf
import sys

filename = sys.argv[1]
print("Extracting table content from:")
print(filename)
# Read tables from the specified PDF file
tables = read_pdf(filename, pages="all")

# Print each table as a DataFrame
for table in tables:
    print(table)
