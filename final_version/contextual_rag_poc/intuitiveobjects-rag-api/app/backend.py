import logging
import threading
import sys
import os

# Add pipeline/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pipeline')))

# Now import
# from pipeline.ingest import process_uploaded_document

from app.pipeline.ingest import process_uploaded_document, logger
# from pipeline.rag_utils import set_active_model, initialize_models, process_query
from app.pipeline.models import get_model_manager
# from pipeline.pdf_to_images import convert_pdf_to_images
from app.pipeline.rag_utils import process_query



from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from gridfs import GridFS
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.logger.setLevel(logging.INFO)



# Initialize models
model_manager = get_model_manager()
model_manager.init_models()  # Initialize models

async def ask_question(question, chat_id=None):
    print('before json')

    try:
        logger.info(f"Received question: {question}")
        response =  await process_query(question, chat_id)
        logger.info(f"Generated response: {response}")

        return response
    except Exception as e:
        logger.error(f'Error processing question: {str(e)}', exc_info=True)
        
        return ({'error': 'An error occurred while processing your question'})



if __name__ == '__main__':
    app.run(host='172.232.105.58', port=5000, debug=True)



    