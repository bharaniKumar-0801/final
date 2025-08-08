import ollama
from typing import List 
import json
async def expand_user_query(conversation: List[dict], user_question: str) -> str:
    """
    Expands a user query into a more detailed, context-aware QUESTION
    using past conversation and Ollama LLM.
    Args:
        conversation (List[dict]): Chat history with 'role' and 'content'
        user_question (str): Latest user question
    Returns:
        str: Expanded query in question format
    """
    # Return the query directly if there's no conversation history
    if not conversation:
        return user_question.strip()
    system_prompt = {
        "role": "system",
        "content": (
            "You are an AI assistant that rephrases vague or short user queries into more detailed, context-rich QUESTIONS.\n"
            "Rules:\n"
            "- ONLY use information present in the conversation history or the query.\n"
            "- DO NOT assume or hallucinate names, entities, or facts that are not explicitly mentioned.\n"
            "- DO NOT add real-world references unless they are already in context.\n"
            "- ONLY return a single well-formed natural language question.\n"
            "- DO NOT explain or generate paragraphs.\n"
            "- Assume the query might refer to internal system or organization-specific terms."
        )
    }
    messages = [system_prompt] + conversation + [
        {"role": "user", "content": f"Expand this query into a well-formed question: '{user_question}'"}
    ]
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=messages,
    )
    return response['message']['content'].strip()


async def metadata_query(expanded_query):
    system_prompt = {
        'role': 'system',
        'content': """
        You are part of an information system that processes users queries.
        Given a user query you extract information from it that matches a given list of metadata fields.
        The information to be extracted from the query must match the semantics associated with the given metadata fields.
        The information that you extracted from the query will then be used as filters to narrow down the search space
        when querying an index.
        Just include the value of the extracted metadata without including the name of the metadata field.
        The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.
        If no information can be extracted from the query, return an empty JSON object.
        """
    }
    user_prompt = {
        'role': 'user',
        'content': expanded_query
    }
    messages = [system_prompt, user_prompt]
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=messages,
    )
    content = response['message']['content'].strip()
    try:
        return json.loads(content)
    except Exception:
        # Optionally log or handle malformed JSON
        return {}