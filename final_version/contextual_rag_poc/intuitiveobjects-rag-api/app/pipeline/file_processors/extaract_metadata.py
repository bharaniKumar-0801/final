# import logging
# from typing import Dict
# import json
# import regex as re
# import requests
# # Example for OpenAI or other LLM integration
# # You can switch to any model like Ollama, Hugging Face, etc.
# import ollama  # Replace or mock this if needed
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# #generate_page_metadata 

# def generate_document_metadata(text: str, model_name: str) -> dict:
#     default_metadata = {
#         "title": "Unknown",
#         "author": "Unknown",
#         "document_owner": "Unknown",
#         "summary": "Metadata generation failed.",
#         "topics": "",
#         "stipend_amount": "Unknown",
#         "mobile_number": "Unknown"
#     }
#     try:
#         prompt = (
#             "You are an intelligent document assistant.\n\n"
#             "Analyze the following document and extract metadata in JSON format with the following fields:\n"
#             "- title: The title of the document (or best inferred)\n"
#             "- author: The issuing organization or company name\n"
#             "- document_owner: The name of the person to whom this document is addressed (e.g., the intern or candidate)\n"
#             "- summary: A detailed 4–6 line summary of the document contents (e.g., what it is, purpose, timeframe, project, policies)\n"
#             "- topics: A list of 4–6 core topics or themes discussed (e.g., Generative AI, internship, project deliverables)\n"
#             "- stipend_amount: The monetary amount of any stipend mentioned \n"
#             "- mobile_number: The mobile number mentioned in the document (in any format)\n\n"
#             ":pushpin: Return ONLY valid JSON. Do not include any explanation, markdown, or extra text.\n"
#             "Your response must be a JSON object with the following keys:\n"
#             "title, author, document_owner, summary, topics, stipend_amount, phone_number\n\n"
#             "Now analyze this document:\n"
#             f"{text}"
#         )
#         # response = ollama.chat(model=model_name, messages=[
#         #     {"role": "user", "content": prompt}
#         # ])

#         response = requests.post(
#             'http://ollama:11434/api/generate',  # Or localhost:11434 if host-based
#             json={
#                 "model": model_name,
#                 "messages": [
#                     {"role": "user", "content": prompt}
#                 ]
#             }
#         )
        



#         raw_content = response.get("message", {}).get("content", "")
#         json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#         else:
#             json_str_match = re.search(r"(\{.*\})", raw_content, re.DOTALL)
#             json_str = json_str_match.group(1) if json_str_match else ""
#         metadata = json.loads(json_str) if json_str else {}
#         logger.info(f"----------------->>>\n{metadata}")

#         # Ensure all keys are present, fill missing with defaults
#         for key, value in default_metadata.items():
#             if key not in metadata or metadata[key] is None:
#                 metadata[key] = value

#         # Convert topics list to comma-separated string if needed
#         if isinstance(metadata.get("topics"), list):
#             metadata["topics"] = ", ".join(metadata["topics"])

#         return metadata

#     except Exception as e:
#         logger.error(f"LLM metadata generation failed: {e}")
#         return default_metadata



# import logging
# from typing import Dict
# import json
# import regex as re
# import requests
# # Example for OpenAI or other LLM integration
# # You can switch to any model like Ollama, Hugging Face, etc.
# import ollama  # Replace or mock this if needed
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# #generate_page_metadata 

# def generate_document_metadata(text: str, model_name: str) -> dict:
#     default_metadata = {
#         "title": "Unknown",
#         "author": "Unknown",
#         "document_owner": "Unknown",
#         "summary": "Metadata generation failed.",
#         "topics": "",
#         "stipend_amount": "Unknown",
#         "mobile_number": "Unknown"
#     }
#     try:
#         prompt = (
#             "You are an intelligent document assistant.\n\n"
#             "Analyze the following document and extract metadata in JSON format with the following fields:\n"
#             "- title: The title of the document (or best inferred)\n"
#             "- author: The issuing organization or company name\n"
#             "- document_owner: The name of the person to whom this document is addressed (e.g., the intern or candidate)\n"
#             "- summary: A detailed 4–6 line summary of the document contents (e.g., what it is, purpose, timeframe, project, policies)\n"
#             "- topics: A list of 4–6 core topics or themes discussed (e.g., Generative AI, internship, project deliverables)\n"
#             "- stipend_amount: The monetary amount of any stipend mentioned \n"
#             "- mobile_number: The mobile number mentioned in the document (in any format)\n\n"
#             ":pushpin: Return ONLY valid JSON. Do not include any explanation, markdown, or extra text.\n"
#             "Your response must be a JSON object with the following keys:\n"
#             "title, author, document_owner, summary, topics, stipend_amount, phone_number\n\n"
#             "Now analyze this document:\n"
#             f"{text}"
#         )
#         # response = ollama.chat(model=model_name, messages=[
#         #     {"role": "user", "content": prompt}
#         # ])

#         response = requests.post(
#             'http://ollama:11434/api/generate',  # Or localhost:11434 if host-based
#             json={
#                 "model": model_name,
#                 "messages": [
#                     {"role": "user", "content": prompt}
#                 ]
#             }
#         )
        



#         raw_content = response.get("message", {}).get("content", "")
#         json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#         else:
#             json_str_match = re.search(r"(\{.*\})", raw_content, re.DOTALL)
#             json_str = json_str_match.group(1) if json_str_match else ""
#         metadata = json.loads(json_str) if json_str else {}
#         logger.info(f"----------------->>>\n{metadata}")

#         # Ensure all keys are present, fill missing with defaults
#         for key, value in default_metadata.items():
#             if key not in metadata or metadata[key] is None:
#                 metadata[key] = value

#         # Convert topics list to comma-separated string if needed
#         if isinstance(metadata.get("topics"), list):
#             metadata["topics"] = ", ".join(metadata["topics"])

#         return metadata

#     except Exception as e:
#         logger.error(f"LLM metadata generation failed: {e}")
#         return default_metadata




import json
import logging
import regex as re
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OLLAMA_BASE_URL = "http://ollama:11434"

def extract_json_from_string(text: str) -> dict:
    try:
        match = re.search(r'{(?:[^{}]|(?R))*}', text)
        if not match:
            raise ValueError("No valid JSON object found in string.")
        return json.loads(match.group(0))
    except Exception as e:
        raise ValueError(f"Failed to extract valid JSON: {e}")

def generate_document_metadata(text: str, model_name: str) -> dict:
    default_metadata = {
        "title": "Unknown",
        "author": "Unknown",
        "document_owner": "Unknown",
        "summary": "Metadata generation failed.",
        "topics": "",
        "stipend_amount": "Unknown",
        "phone_number": "Unknown"
    }

    try:
        model_name = model_name.strip()  # Clean up accidental trailing space

        prompt = (
            "You are an intelligent document assistant.\n\n"
            "Analyze the following document and extract metadata in JSON format with the following fields:\n"
            "- title: The title of the document (or best inferred)\n"
            "- author: The issuing organization or company name\n"
            "- document_owner: The name of the person to whom this document is addressed (e.g., the intern or candidate)\n"
            "- summary: A detailed 4–6 line summary of the document contents (e.g., what it is, purpose, timeframe, project, policies)\n"
            "- topics: A list of 4–6 core topics or themes discussed (e.g., Generative AI, internship, project deliverables)\n"
            "- stipend_amount: The monetary amount of any stipend mentioned \n"
            "- phone_number: The phone number mentioned in the document (in any format)\n\n"
            ":pushpin: Return ONLY valid JSON. Do not include any explanation, markdown, or extra text.\n"
            "Your response must be a JSON object with the following keys:\n"
            "title, author, document_owner, summary, topics, stipend_amount, phone_number\n\n"
            "Now analyze this document:\n"
            f"Document:\n{text}"
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        # logger.info(f"Raw LLM response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Ollama error {response.status_code}: {response.text}")

        response_data = response.json()
        content = response_data.get("response", "").strip()

        metadata = extract_json_from_string(content)

        # Ensure all required fields are present
        for key in default_metadata:
            if key not in metadata or not metadata[key]:
                metadata[key] = default_metadata[key]

        if isinstance(metadata.get("topics"), list):
            metadata["topics"] = ", ".join(map(str, metadata["topics"]))

        logger.info(f"Extracted metadata: {metadata}")
        return metadata

    except Exception as e:
        logger.error(f"LLM metadata generation failed: {e}")
        return default_metadata