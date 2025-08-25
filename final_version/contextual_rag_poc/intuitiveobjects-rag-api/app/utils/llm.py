# # import ollama
# # from typing import List 
# # import json


# # async def expand_user_query(conversation: List[dict], user_question: str) -> str:
# #     """
# #     Expands a user query into a more detailed, context-aware QUESTION 
# #     using past conversation and Ollama LLM.

# #     Args:
# #         conversation (List[dict]): Chat history with 'role' and 'content'
# #         user_question (str): Latest user question

# #     Returns:
# #         str: Expanded query in question format
# #     """
# #     # Return the query directly if there's no conversation history
# #     if not conversation:
# #         return user_question.strip()

# #     system_prompt = {
# #         "role": "system",
# #         "content": (
# #             "You are an AI assistant that rephrases vague or short user queries into more detailed, context-rich QUESTIONS.\n"
# #             "Rules:\n"
# #             "- ONLY use information present in the conversation history or the query.\n"
# #             "- DO NOT assume or hallucinate names, entities, or facts that are not explicitly mentioned.\n"
# #             "- DO NOT add real-world references unless they are already in context.\n"
# #             "- ONLY return a single well-formed natural language question.\n"
# #             "- DO NOT explain or generate paragraphs.\n"
# #             "- Assume the query might refer to internal system or organization-specific terms."
# #         )
# #     }

# #     messages = [system_prompt] + conversation + [
# #         {"role": "user", "content": f"Expand this query into a well-formed question: '{user_question}'"}
# #     ]

# #     response = ollama.chat(
# #         model="qwen2.5:1.5b",
# #         messages=messages,
# #     )



# #     return response['message']['content'].strip()














# # async def metadata_query(expanded_query):
# #     system_prompt = {
# #         'role': 'system',
# #         'content': """
# #         You are part of an information system that processes users queries.
# #         Given a user query you extract information from it that matches a given list of metadata fields.
# #         The information to be extracted from the query must match the semantics associated with the given metadata fields.
# #         The information that you extracted from the query will then be used as filters to narrow down the search space
# #         when querying an index.
# #         Just include the value of the extracted metadata without including the name of the metadata field.
# #         The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.
# #         If no information can be extracted from the query, return an empty JSON object.
# #         """
# #     }
# #     user_prompt = {
# #         'role': 'user',
# #         'content': expanded_query
# #     }
# #     messages = [system_prompt, user_prompt]
# #     response = ollama.chat(
# #         model="qwen2.5:1.5b",
# #         messages=messages,
# #     )
# #     content = response['message']['content'].strip()
# #     try:
# #         return json.loads(content)
# #     except Exception:
# #         # Optionally log or handle malformed JSON
# #         return {} 
    

import ollama
from typing import List, Dict, Any
import json
import logging
import re
import requests 
import time

logger = logging.getLogger(__name__) 


from app.pipeline.models import get_model_manager
model_manager = get_model_manager()
model_name = model_manager.initialize_qwen3_8b_model()
# model_name = model_manager.initialize_qwen_model()

def remove_think_tag(text: str) -> str:
    """Remove <think>...</think> blocks from the text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL) 


def call_ollama_generate(payload, max_retries=2, wait_seconds=2):
    for attempt in range(max_retries):
        response = requests.post(
            "http://ollama:11434/api/generate",
            json=payload
        )
        result_json = response.json()
        # If response is not empty or done_reason is not "load", return it
        if result_json.get("response", "").strip() or result_json.get("done_reason") != "load":
            return result_json
        time.sleep(wait_seconds)  # Wait before retrying
    return result_json  # Return last response even if empty
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

  
    # system_prompt = {
    #         "role": "system",
    #     "content": (
    #         "You are an AI assistant that expands vague or ambiguous user queries into detailed, context-rich questions.\n"
    #         "You MUST use all relevant information from the conversation history, including project names, entities, or details mentioned in previous user queries and assistant responses.\n"
    #         "If the conversation history contains the answer to an ambiguous part of the current query (such as a project name), use it to make the expanded question specific and complete.\n"
    #         "NEVER ask the user for more context if it is already present in the conversation history.\n"
    #         "NEVER repeat or rephrase the user's request for clarification if you can resolve it from context.\n"
    #         "ONLY use information present in the conversation history or the current query.\n"
    #         "DO NOT assume or hallucinate names, entities, or facts that are not explicitly mentioned.\n"
    #         "Return a single, well-formed natural language question that is as specific as possible using all available context.\n"
    #         "DO NOT explain or generate paragraphs."
    #     )
    #     } 

    system_prompt = {
            "role": "system",
            "content": (
                "You are an AI assistant that expands vague or ambiguous user queries into detailed, context-rich questions.\n"
                "If the user query is already a clear, well-formed question (e.g., starts with 'who', 'what', 'when', 'where', 'why', or 'how'), KEEP it as-is and do not transform it into a yes/no question.\n"
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

    logger.info(f"[expand_user_query] Messages sent to Ollama: {messages}")

    response = requests.post(
        "http://ollama:11434/api/chat",
        json={
            "model": model_name,
            "messages": messages,
            "stream": False
        },
        
    )

    result_json = response.json()
    logger.info(f"[expand_user_query] Ollama raw response: {json.dumps(result_json, indent=2)}")

   
  
    raw_content = (
        result_json.get("response") or
        result_json.get("message", {}).get("content") or
        result_json.get("output") or
        result_json.get("data", [{}])[0].get("content", "") or
        ""
    ).strip()

    expanded_result = remove_think_tag(raw_content)
    logger.info(f"[expand_user_query] Expanded query: {expanded_result}")
    return expanded_result


async def metadata_query(expanded_query: str) -> Dict[str, Any]:
    """
    Extracts metadata from an expanded query using Ollama LLM.
    Returns a dictionary of extracted metadata.
    """
    system_instructions = (
        "You are part of an information system that processes users queries.\n"
        "Given a user query you extract information from it that matches a given list of metadata fields.\n"
        "The information to be extracted from the query must match the semantics associated with the given metadata fields.\n"
        "The information that you extracted from the query will then be used as filters to narrow down the search space "
        "when querying an index.\n"
        "Just include the value of the extracted metadata without including the name of the metadata field.\n"
        "The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.\n"
        "If no information can be extracted from the query, return an empty JSON object."
    )

    # Build prompt for generate
    prompt = f"""{system_instructions}

User query:
{expanded_query}

Return only valid JSON.
"""

    logger.debug(f"[metadata_query] Prompt sent to Ollama:\n{prompt}")

    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        },
        
    )

    result_json = response.json()
    raw_content = result_json.get("response", "").strip()

    logger.info(f"[metadata_query] Raw response: {raw_content}")

    try:
        metadata = json.loads(raw_content)
        logger.info(f"[metadata_query] Parsed metadata: {metadata}")
        return metadata
    except Exception as e:
        logger.warning(f"[metadata_query] Failed to parse JSON: {e}. Content: {raw_content}")
        return {}


# from keybert import KeyBERT

# kw_model = KeyBERT()

# def extract_keywords(text):
#     keywords = kw_model.extract_keywords(
#         text,
#         keyphrase_ngram_range=(1, 3),  # single word to 3-word phrases
#         stop_words='english',
#         top_n=5
#     )
#     return [kw[0] for kw in keywords]



# import spacy
# from keybert import KeyBERT

# # Load spaCy model once
# nlp = spacy.load("en_core_web_sm")


# class KeywordExtractor:
#     def __init__(self):
#         self.kw_model = KeyBERT()
#         self.stopwords = spacy.lang.en.stop_words.STOP_WORDS

#     def preprocess(self, query: str):
#         """Tokenize, remove stopwords/punct, lemmatize"""
#         doc = nlp(query.lower())
#         tokens = [
#             token.lemma_
#             for token in doc
#             if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "PROPN"}
#         ]
#         return tokens

#     def extract_phrases(self, query: str):
#         """Extract multi-word noun phrases"""
#         doc = nlp(query)
#         phrases = [
#             chunk.text.lower()
#             for chunk in doc.noun_chunks
#             if len(chunk.text.split()) > 1
#         ]
#         return phrases

#     def extract_keywords(self, query: str, top_n: int = 10):
#         """Rank keywords with KeyBERT; fallback to raw candidates if empty"""

#         logger.info(f"[KeywordExtractor] Extracting keywords from query: {query}")
#         # Deduplicate while preserving order
#         candidates = list(dict.fromkeys(self.preprocess(query) + self.extract_phrases(query)))

#         if not candidates:
#             return []

#         # Try ranking with KeyBERT
#         ranked = self.kw_model.extract_keywords(
#             query,
#             candidates=candidates,
#             keyphrase_ngram_range=(1, 3),
#             stop_words="english",
#             top_n=top_n,
#         )

#         if ranked:
#             return [kw for kw, score in ranked]
#         else:
#             return candidates[:top_n]


# # ------------------------------
# # Example usage
# # ------------------------------
# if __name__ == "__main__":
#     extractor = KeywordExtractor()



#     keywords = extractor.extract_keywords(query, top_n=10) 
    
#     logger.info(f"Ranked Keywords: {keywords}")



import spacy
from keybert import KeyBERT

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def preprocess(self, query: str):
        """Tokenize, remove stopwords/punct, lemmatize"""
        doc = nlp(query.lower())

        # First try to keep only nouns/proper nouns
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "PROPN"}
        ]

        # Fallback: if nothing extracted, keep all alphabetic words
        if not tokens:
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]

        return tokens

    def extract_phrases(self, query: str):
        """Extract multi-word noun phrases"""
        doc = nlp(query)
        phrases = [
            chunk.text.lower()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) > 1
        ]
        return phrases

    def extract_keywords(self, query: str, top_n: int = 10):
        """Rank keywords with KeyBERT; fallback to raw candidates if empty"""
        candidates = list(dict.fromkeys(self.preprocess(query) + self.extract_phrases(query)))

        if not candidates:
            return []

        ranked = self.kw_model.extract_keywords(
            query,
            candidates=candidates,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
        )

        if ranked:
            return [kw for kw, score in ranked]
        else:
            return candidates[:top_n]
