


import os
import pickle
import logging
from typing import List, Dict
import re
from app.services.app_config_service import get_app_configs
from app.pipeline.models import get_model_manager, llm_generate_response
from app.pipeline.rag_pipeline import similarity_search
from app.pipeline.keyword_search.text_utils import enrich_tokens
from app.pipeline.keyword_search.bm25_engine import BM25
# from app.services.chat_service import metadata_query 
# from app.services.chat_service import expand_user_query
from app.pipeline.rag_pipeline import expand_query_with_context 
from app.utils.llm import metadata_query
from sentence_transformers import CrossEncoder
# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Load model manager globally
model_manager = get_model_manager()

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def remove_markdown(text: str) -> str:
    # Remove asterisks used for bold/italic
    return re.sub(r"\*+", "", text)

async def process_query(query: str, chat_id: str) -> str:
    """
    Handles the full RAG pipeline:
    1. Loads configs and sets the active model.
    2. Retrieves top results from ChromaDB.
    3. Performs keyword-based BM25 search from disk.
    4. Combines results and sends enriched query to the LLM.
    """
    try:
        # 1. Load configuration
        app_config = await get_app_configs()
        model_name = app_config.get("llm_model")
        temperature = app_config.get("temperature", 0.7)

        # model_manager.set_active_model(model_name)
        # active_model = model_manager.get_active_model()

        expanded_query = await expand_query_with_context(chat_id, query) 

        logger.info(f"[process_query] expanded_query: {expanded_query}")

        metadata_query_result = await enrich_tokens(expanded_query)

        logger.info(f"[process_query] Metadata query: {metadata_query_result}")

        logger.info(f"[process_query] Metadata query result: {metadata_query_result}")

        model_manager.set_active_model(model_name)
        active_model = model_manager.get_active_model()
        # 2. Vector Search (Chroma)
        vector_results = similarity_search(query)[:5]
        logger.info(f"Vector search found {vector_results} results.")

        def metadata_match(chunk, keywords):
            metadata_values = " ".join(str(v).lower() for v in chunk.get("metadata", {}).values())
            return any(str(kw).lower() in metadata_values for kw in keywords)

        filtered_vector_results = [
            chunk for chunk in vector_results
            if metadata_match(chunk, metadata_query_result)
        ]
        logger.info(f"Filtered vector results: {filtered_vector_results}")

        # # 3. BM25 Keyword Search
        # bm25_chunks = await run_bm25_keyword_search(query, top_n=5) 
        # logger.info(f"BM25 search returned {bm25_chunks} results.")

        # # 4. Combine results
        # top_chunks = filtered_vector_results + bm25_chunks
        # logger.info(f"Combined top chunks: {top_chunks} results.") 

  

        # ranked_chunks = rerank_results(query, top_chunks, top_k=5)
        # logger.info(f"Reranked top chunks: {ranked_chunks} results.")

        # # Build context: vector results with metadata, BM25 with just text
        # context_lines = []
        # for chunk in ranked_chunks:
        #     text = chunk["text"]
        #     metadata = chunk.get("metadata")
        #     if metadata:
        #         # Vector result: include metadata
        #         context_lines.append(f"{text}\nMetadata: {metadata}")
        #     else:
        #         # BM25 result: only text
        #         context_lines.append(text)
        # context = "\n".join(context_lines) 

        # context = remove_markdown(context) 


        enriched_query = f"Answer the query using the context and metadata.\n\nQuery:\n{query}\nContext:\n{filtered_vector_results}"

        logger.debug(f"Enriched query:\n{enriched_query}")

        # 5. Generate response
        response = llm_generate_response(active_model, enriched_query, temperature=temperature)
        response = remove_markdown(response)
        return response

    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
        return "I'm sorry, I encountered an error while processing your query. Please try again later." 

async def run_bm25_keyword_search(query: str, top_n: int = 10) -> List[Dict]:
    """
    Performs keyword-based BM25 search using saved indexes from disk.
    Returns top-N results globally across all indexes.
    """
    bm25_dir = "./app/pipeline/bm25_store"
    bm25_chunks = []

    if not os.path.isdir(bm25_dir):
        logger.warning(f"BM25 directory not found: {bm25_dir}")
        return []

    keyword_tokens = await enrich_tokens(query)
    logger.debug(f"BM25 extracted keywords: {keyword_tokens}")

    for filename in os.listdir(bm25_dir):
        if not filename.endswith(".pkl"):
            continue

        file_path = os.path.join(bm25_dir, filename)

        try:
            with open(file_path, "rb") as f:
                # Unpack saved dict from ingest_text.py
                data = pickle.load(f)
                bm25_index: BM25 = data["bm25"]
                # metadata = data["metadata"]

            top_results = bm25_index.get_top_n(keyword_tokens, n=top_n)

            logger.info(f"BM25 search in {filename} returned {len(top_results)} results.")

            for idx, score in top_results:
                chunk_text = " ".join(bm25_index.tokenized_docs[idx])
                bm25_chunks.append({
                    "text": chunk_text,
                    "score": score,
                    "chunk_id": f"bm25_{filename}_{idx}",
                    # "metadata": metadata[idx] if isinstance(metadata, list) and idx < len(metadata) else {}
                })

        except Exception as e:
            logger.warning(f"Failed to process BM25 index for {filename}: {e}")

    sorted_chunks = sorted(bm25_chunks, key=lambda x: x["score"], reverse=True)[:top_n]
    logger.info(f"BM25 global top-{top_n} results selected.")
    return sorted_chunks





def rerank_results(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Reranks chunks using a cross-encoder model based on relevance to the query.
    Filters out chunks below min_score if specified.
    """
    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker_model.predict(pairs)

    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # if min_score is not None:
    #     sorted_chunks = [chunk for chunk in sorted_chunks if chunk["rerank_score"] >= min_score]

    return sorted_chunks[:top_k]