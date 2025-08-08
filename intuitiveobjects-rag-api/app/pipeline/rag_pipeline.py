import chromadb
import ollama
from typing import List, Dict
from chromadb.utils import embedding_functions

import os

from .models import get_model_manager 
from .bm25.bm25_impl import BM25Index
from app.utils.llm import expand_user_query 
# Set up loggings
import logging
from .logger_config import setup_logger
logger = setup_logger(__name__, log_level=logging.INFO) 

# Initialize ChromaDB client

# PERSIST_DIRECTORY = os.path.join(os.getcwd(), "/home/vishwa/harry_rag/intuitiveobjects-rag-api/chroma_storage")
# os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

PERSIST_DIRECTORY = "/home/bharani/final_integration/intuitiveobjects-rag-api/chroma_storage"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

#chroma_client = chromadb.Client()

# After adding documents, you can verify the collection exists
collection_names = chroma_client.list_collections()
logger.info(f"Available collections: {collection_names}")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# collection = chroma_client.get_or_create_collection(
#     name="document_collection",
#     embedding_function=embedding_function
# )
# Print the embedding dimension
test_embedding = embedding_function(["test"])
embedding_dim = len(test_embedding[0])
logger.info(f"Embedding dimension in rag_pipeline: {embedding_dim}")

model_manager = get_model_manager()

bm25_index = BM25Index("./bm25_index_store", True)

def vector_similarity_search(query: str, n_results: int = 5) -> List[Dict]:
    
    """
    Perform a similarity search in ChromaDB.
    """
    # Convert the query to an embedding
    query_embedding = embedding_function([query])[0]
    logger.info(f"Query embedding dimension: {len(query_embedding)}")
    results = collection.query(
        query_embeddings=[query_embedding],  # Use query_embeddings instead of query_texts
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    logger.info(f"Found {len(results['documents'][0])} matching documents")
    return list(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]))



# def similarity_search(query: str, n_results: int = 3) -> List[Dict]:
#     """
#     Perform a similarity search in ChromaDB.
#     """
#     # Convert the query to an embedding
#     query_embedding = embedding_function([query])[0]
#     logger.info(f"Query embedding dimension: {len(query_embedding)}")
#     results = collection.query(
#         query_embeddings=[query_embedding],  # Use query_embeddings instead of query_texts
#         n_results=n_results,
#         include=['documents', 'metadatas']
#     )
#     logger.info(f"Type of ChromaDB results = {type(results)}")
#     logger.info(f"{results}")
#     logger.info(f"Found {len(results['documents'][0])} matching documents")
#     return list(zip(results['documents'][0], results['metadatas'][0]))


def similarity_search(query_text, filters=None, top_k=10):
    try:
        logger.info(f"Available collections: {collection_names}")



        collection = chroma_client.get_or_create_collection(
    name="document_collection",
    embedding_function=embedding_function
)
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filters or {}  # optional metadata filter
        ) 

        logger.info(f"ChromaDB query results: {results}")

        documents = results.get("documents", [[]])[0]       # list of matched texts
        metadatas = results.get("metadatas", [[]])[0]       # list of metadata dicts
        ids = results.get("ids", [[]])[0]                   # chunk IDs    
        logger.info(f"Documents: {documents}")
        logger.info(f"Metadatas: {metadatas}")
        # logger.info(f"metadatas: {metadatas}")
   

        combined = [
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i]
            }
            for i in range(len(documents))
        ]

        # logger.info('combined>>>>>>>>>>>', combined)
        return combined

    except Exception as e:
        logger.error(f"Error during ChromaDB query: {e}")
        return []




# def enrich_query_context(query: str, search_results: List[Dict]) -> str:
#     """
#     Enrich the query context with search results.
#     """
#     #context = "You are a helpful AI assistant. Use the following context to answer the user's question.\n\nContext:\n"
#     context = "Context:\n"
#     for doc, metadata in search_results:
#         context += f"- {doc}\n"
#     context += f"\nQuestion: {query}\n"
#     context += "\nAnswer the question based on the context provided. If the answer cannot be found in the context, say 'I don't have enough information to answer that question.'\n"
#     logger.info(f"Enriched query context: {context}")
#     return context


# def enrich_query_context(query: str, search_results: List[Dict]) -> str:
#     """
#     Enrich the query context with search results.
#     """
#     context = "Context:\n"
#     for result in search_results:
#         doc = result.get("text", "")
#         metadata = result.get("metadata", {})

#         logger.info(f"metadata: {metadata}")


#         context += f"- {doc}\n"
#     context += f"\nQuestion: {query}\n"
#     context += "\nAnswer the question based on the context provided. If the answer cannot be found in the context, say 'I don't have enough information to answer that question.'\n"
#     logger.info(f"Enriched query context: {context}")
#     return context



def enrich_query_context(query: str, search_results: List[Dict]) -> str:
    """
    Enrich the query context with search results and their metadata.
    """
    context = "You are a helpful AI assistant. Use the following context (including Document Chunk and metadata) to answer the user's question.\n\nContext:\n"

    for result in search_results:
        doc = result.get("text", "")
        metadata = result.get("metadata", {})

        # Format metadata nicely
        formatted_metadata = "\n".join([f"  {key}: {value}" for key, value in metadata.items()])
        
        context += f"- Document Chunk:\n{doc}\nMetadata:\n{formatted_metadata}\n\n"

    context += f"Question: {query}\n"
    context += "\nAnswer the question based on the above context. If the answer is not available, respond with 'I don't have enough information to answer that question.'\n"

    logger.info(f"Enriched query context: {context}")
    return context






def print_hybrid_search_results(top_k_results):
    if len(top_k_results) == 0:
        logger.info(f"top_k_results is empty")
    else: 
        print(top_k_results)
    return

def hybrid_search(
                 query: str, 
                 k: int = 5, 
                 alpha: float = 0.5
                 ) -> List[Dict[str, any]]:
    """
    Perform hybrid search using both BM25 and vector similarity
    
    Args:
        query: Search query
        k: Number of results to return
        alpha: Weight for combining scores (0 = only BM25, 1 = only vector)
        
    Returns:
        List of results with combined scores
    """
    logger.info(f"Performing hybrid search for query: '{query}'")
    
    # Get vector search results
    vector_results = vector_similarity_search(query, k)
   
    # Debug the structure
    logger.info(f"Vector results type: {type(vector_results)}")
    logger.info(f"Vector results structure: {vector_results}")

    # Get BM25 results
    bm25_results = bm25_index.search(query, top_k=k)
    
    # Combine results (simplified version - you might want to implement
    # a more sophisticated combination strategy)
    combined_results = []
    seen_chunks = set()
    
    # Process vector results
    for doc, metadata, similarity_score in vector_results:
        if doc not in seen_chunks:
            seen_chunks.add(doc)
            combined_results.append({
                    'chunk': doc,
                    'metadata': metadata,
                    'vector_score': similarity_score,  # Simple score since distances aren't available
                    'bm25_score': 0.0  # Will be updated if found in BM25 results
                })

    logger.info(f"hybrid_search: Found {len(vector_results)} vector matches")

    """"""
    # if 'documents' in vector_results and vector_results['documents']:
        # documents = vector_results['documents'][0]  # First list in documents
        # metadatas = vector_results['metadatas'][0]  # First list in metadatas
        # distances = vector_results['distances'][0]  # First list in distances
        # 
        # logger.info(f"Found {len(documents)} vector matches")
        
        # Add vector results to combined results
        # for doc, metadata,distance in zip(documents, metadatas, distances):
            # if doc not in seen_chunks:
                # seen_chunks.add(doc)
                # combined_results.append({
                    # 'chunk': doc,
                    # 'metadata': metadata,
                    # 'vector_score': distance,  # Simple score since distances aren't available
                    # 'bm25_score': 0.0  # Will be updated if found in BM25 results
                # })
    # else:
        # logger.warning("No vector search results found...but why ???")
    """"""

    # Process BM25 results
    for doc_id, doc, score in bm25_results:
        if doc not in seen_chunks:
            seen_chunks.add(doc)
            combined_results.append({
                'chunk': doc,
                'metadata': {'source': 'bm25_only'},
                'vector_score': 0.0,
                'bm25_score': score
            })
        else:
            # Update BM25 score for existing result
            for result in combined_results:
                if result['chunk'] == doc:
                    result['bm25_score'] = score
                    break
    
    # Calculate combined scores
    for result in combined_results:
        result['combined_score'] = (
            alpha * result['vector_score'] + 
            (1 - alpha) * result['bm25_score']
        )
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    logger.info(f"Found {len(combined_results)} results")
    top_k = combined_results[:k]
    return top_k



def generate_response(enriched_query: str) -> str:
    """
    Generate a response using Ollama 8B model.
    """
    response = ollama.generate(
        model='llama2:7b',
        prompt=enriched_query,
        options={
            'temperature': 1,
            'num_predict': 500,  # This replaces max_tokens
        }
    )
    logger.info(f"Raw Ollama response: {response}")
    return response['response']


"""
def process_query(query: str) -> str:
    #Process a user query through the RAG pipeline.
    try:
        search_results = similarity_search(query)
        enriched_query = enrich_query_context(query, search_results)
        # Generate response using the active model
        active_model, active_tokenizer = model_manager.get_active_model()
        response = llm_generate_response(active_model, active_tokenizer, enriched_query)
        #response = generate_response(enriched_query)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return "I'm sorry, but I encountered an error while processing your query. Please try again later."
"""

# Function to set the active model
def set_active_model(model_name):
    model_manager.set_active_model(model_name)

# Initialize the models
def initialize_models():
    model_manager.init_models()

""""""
# if __name__ == "__main__":
    # Test the pipeline
    # test_query = "What is the capital of France?"
    # result = process_query(test_query)
    # print(f"Query: {test_query}")
    # print(f"Response: {result}")
""""""


from app.db.mongodb import user_query_collection  # Adjust import path if needed
logger = logging.getLogger(__name__)
async def expand_query_with_context(chat_id: str, user_query: str) -> str:
    try:
        # Fetch messages ordered by creation time
        messages = await user_query_collection().find(
            {"chat_id": chat_id}
        ).sort("created_at", 1).to_list(length=100)
        # Convert messages into list of dicts with 'role' and 'content'
        conversation = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg.get("role") in ["user", "assistant"]
        ]
        # Log the conversation type and preview of content
        logger.info(f"[expand_query_with_context] Conversation type: {type(conversation)}")
        logger.debug(f"[expand_query_with_context] First 2 messages: {conversation[:2]}")
        # Call the LLM to expand the user query
        expanded_query = await expand_user_query(conversation, user_query) 
        # metadata_query_result = metadata_query(expanded_query)
        # logger.info(f"[expand_query_with_context] Metadata query result: {metadata_query_result}")
        logger.info(f"[expand_query_with_context] Expanded query: {expanded_query}")
        return expanded_query
    except Exception as e:
        logger.error(f"[expand_query_with_context] Failed to expand query: {str(e)}", exc_info=True)
        return user_query  # fallback to original query if expansion fails