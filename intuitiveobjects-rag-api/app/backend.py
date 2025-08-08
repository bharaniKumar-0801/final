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

# # MongoDB connection
# client = MongoClient('mongodb://172.235.25.175:5000')
# db = client['database_name']
# fs = GridFS(db)  # Initialize GridFS

# Initialize models
model_manager = get_model_manager()
model_manager.init_models()  # Initialize models

# #store llm model in db
# @app.route('/organization-admin/organization-app-config',methods=['PUT'])
# def update_app_config():
#     config= request.json
#     db['organization_app_config'].update_one(
#         {'organization_id': config["organization_id"]},
#         {'$set': config},   
#         upsert=True
#     )
#     return jsonify({'message': 'App config updated successfully'}), 200

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         app.logger.error('No file part in the request')
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         app.logger.error('No selected file')
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file:
#         try:
#             filename = file.filename
#             content = file.read()
            
#             app.logger.info(f'Attempting to upload file: {filename}')
            
#             # Store file in MongoDB using GridFS
#             file_id = fs.put(io.BytesIO(content), filename=filename)
            
#             app.logger.info(f'File uploaded successfully. File ID: {file_id}')

#              # Process the document asynchronously
#             threading.Thread(target=process_uploaded_document, args=(file_id,)).start()
#             return jsonify({'message': 'File uploaded successfully', 'file_id': str(file_id)}), 200
#         except Exception as e:
#             app.logger.error(f'Error uploading file: {str(e)}')
#             return jsonify({'error': str(e)}), 500


    # Placeholder for RAG pipeline
    # question = request.json.get('question')
    # Process the question using your RAG pipeline here
    # answer = f"This is a placeholder answer for the question: {question}"
    # return jsonify({'answer': answer}), 200

    ##
# def ask_question():
# @app.route('/ask', methods=['POST'])
async def ask_question(question, chat_id=None):
    print('before json')
    # data = request.json
    # print('data>>>>>>>',data)
    # if 'question' not in data:
    #     return jsonify({'error': 'No question provided'}), 400
    
    # question = data['question']
    # logger.info(f"Received question: {question}")
    try:
        response =  await process_query(question, chat_id)
        logger.info(f"Generated response: {response}")
        # response_json = {'response': response}
        # logger.info(f"Sending JSON response: {response_json}")
        # return jsonify(response_json), 200
        # return response_json
        return response
    except Exception as e:
        logger.error(f'Error processing question: {str(e)}', exc_info=True)
        # return jsonify({'error': 'An error occurred while processing your question'}), 500
        return ({'error': 'An error occurred while processing your question'})

    ##

if __name__ == '__main__':
    app.run(host='172.235.25.175', port=5000, debug=True)



    