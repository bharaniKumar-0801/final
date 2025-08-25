import magic

def detect_file_type(file_content):
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file_content)
    return file_type
