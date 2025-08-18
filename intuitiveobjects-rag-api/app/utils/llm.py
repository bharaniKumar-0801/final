# import ollama
# from typing import List 
# import json


# async def expand_user_query(conversation: List[dict], user_question: str) -> str:
#     """
#     Expands a user query into a more detailed, context-aware QUESTION 
#     using past conversation and Ollama LLM.

#     Args:
#         conversation (List[dict]): Chat history with 'role' and 'content'
#         user_question (str): Latest user question

#     Returns:
#         str: Expanded query in question format
#     """
#     # Return the query directly if there's no conversation history
#     if not conversation:
#         return user_question.strip()

#     system_prompt = {
#         "role": "system",
#         "content": (
#             "You are an AI assistant that rephrases vague or short user queries into more detailed, context-rich QUESTIONS.\n"
#             "Rules:\n"
#             "- ONLY use information present in the conversation history or the query.\n"
#             "- DO NOT assume or hallucinate names, entities, or facts that are not explicitly mentioned.\n"
#             "- DO NOT add real-world references unless they are already in context.\n"
#             "- ONLY return a single well-formed natural language question.\n"
#             "- DO NOT explain or generate paragraphs.\n"
#             "- Assume the query might refer to internal system or organization-specific terms."
#         )
#     }

#     messages = [system_prompt] + conversation + [
#         {"role": "user", "content": f"Expand this query into a well-formed question: '{user_question}'"}
#     ]

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=messages,
#     )



#     return response['message']['content'].strip()














# async def metadata_query(expanded_query):
#     system_prompt = {
#         'role': 'system',
#         'content': """
#         You are part of an information system that processes users queries.
#         Given a user query you extract information from it that matches a given list of metadata fields.
#         The information to be extracted from the query must match the semantics associated with the given metadata fields.
#         The information that you extracted from the query will then be used as filters to narrow down the search space
#         when querying an index.
#         Just include the value of the extracted metadata without including the name of the metadata field.
#         The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.
#         If no information can be extracted from the query, return an empty JSON object.
#         """
#     }
#     user_prompt = {
#         'role': 'user',
#         'content': expanded_query
#     }
#     messages = [system_prompt, user_prompt]
#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=messages,
#     )
#     content = response['message']['content'].strip()
#     try:
#         return json.loads(content)
#     except Exception:
#         # Optionally log or handle malformed JSON
#         return {} 
    

import ollama
from typing import List, Dict, Any
import json
import logging
import re

logger = logging.getLogger(__name__) 




def remove_think_tag(text: str) -> str:
    """Remove <think>...</think> blocks from the text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

async def expand_user_query(conversation: List[Dict[str, Any]], user_question: str) -> str:
    """
    Expands a user query into a more detailed, context-aware QUESTION 
    using past conversation and Ollama LLM.

    Args:
        conversation (List[dict]): Chat history with 'role' and 'content'
        user_question (str): Latest user question

    Returns:
        str: Expanded query in question format
    """
    # Ensure conversation is a list of dicts
    if not isinstance(conversation, list):
        logger.warning("Conversation is not a list, converting to list.")
        conversation = [{"role": "user", "content": str(conversation)}]

    # Return the query directly if there's no conversation history
    if not conversation:
        return user_question.strip()

    system_prompt = {
        "role": "system",
    "content": (
        "You are an AI assistant that expands vague or ambiguous user queries into detailed, context-rich questions.\n"
        "You MUST use all relevant information from the conversation history, including project names, entities, or details mentioned in previous user queries and assistant responses.\n"
        "If the conversation history contains the answer to an ambiguous part of the current query (such as a project name), use it to make the expanded question specific and complete.\n"
        "NEVER ask the user for more context if it is already present in the conversation history.\n"
        "NEVER repeat or rephrase the user's request for clarification if you can resolve it from context.\n"
        "ONLY use information present in the conversation history or the current query.\n"
        "DO NOT assume or hallucinate names, entities, or facts that are not explicitly mentioned.\n"
        "Return a single, well-formed natural language question that is as specific as possible using all available context.\n"
        "DO NOT explain or generate paragraphs."
    )
    }

    messages = [system_prompt] + conversation + [
        {"role": "user", "content": f"Expand this query into a well-formed question: '{user_question}'"}
    ]

    logger.debug(f"[expand_user_query] Messages sent to Ollama: {messages}")

    response = ollama.chat(
        model="qwen3:8b",
        messages=messages,
    )

    expanded = response['message']['content'].strip()
    expanded_result = remove_think_tag(expanded)
    logger.info(f"[expand_user_query] Expanded query: {expanded_result}")
    return expanded_result

async def metadata_query(expanded_query: str) -> Dict[str, Any]:
    """
    Extracts metadata from an expanded query using Ollama LLM.
    Returns a dictionary of extracted metadata.
    """
    system_prompt = {
        'role': 'system',
        'content': (
            "You are part of an information system that processes users queries.\n"
            "Given a user query you extract information from it that matches a given list of metadata fields.\n"
            "The information to be extracted from the query must match the semantics associated with the given metadata fields.\n"
            "The information that you extracted from the query will then be used as filters to narrow down the search space "
            "when querying an index.\n"
            "Just include the value of the extracted metadata without including the name of the metadata field.\n"
            "The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.\n"
            "If no information can be extracted from the query, return an empty JSON object."
        )
    }
    user_prompt = {
        'role': 'user',
        'content': expanded_query
    }
    messages = [system_prompt, user_prompt]

    logger.debug(f"[metadata_query] Messages sent to Ollama: {messages}")

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=messages,
    )
    content = response['message']['content'].strip()
    logger.info(f"[metadata_query] Raw response: {content}")
    try:
        metadata = json.loads(content)
        logger.info(f"[metadata_query] Parsed metadata: {metadata}")
        return metadata
    except Exception as e:
        logger.warning(f"[metadata_query] Failed to parse JSON: {e}. Content: {content}")