from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path
import os
import time
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')

# Add handler (console in this case)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

class BM25Index:
    def __init__(self, index_dir: str, auto_persist: bool = True):
        """
        Initialize the BM25 searcher with index directory
        
        Args:
            index_dir: Directory where the BM25 index will be stored
            auto_persist: If True, saves index after each modification. 
                         If False, requires manual calls to save_index()
        """
        # Ensure the directory path is absolute
        self.index_dir = os.path.abspath(index_dir)
        self.index_path = os.path.join(self.index_dir, "bm25_index.pkl")
        self.auto_persist = auto_persist
        
        logger.info(f"Initializing BM25 searcher with index directory: {self.index_dir}")
        logger.info(f"Auto-persist is {'enabled' if auto_persist else 'disabled'}")
        
        # Create directory if it doesn't exist
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        
        self.bm25 = None
        self.documents = {}
        self.tokenized_docs = {}
        self.modified = False
        self.load_index()
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess and tokenize text"""
        tokens = text.lower().split()
        logger.debug(f"Preprocessed text into {len(tokens)} tokens")
        return tokens
    
    def add_document(self, doc_id: str, document: str, persist: bool = None) -> None:
        """
        Add a single document to the index
        
        Args:
            doc_id: Unique identifier for the document
            document: Text content of the document
            persist: Override auto_persist setting for this operation
        """
        logger.info(f"Adding document with ID: {doc_id}")
        
        # Store the document and its tokens
        self.documents[doc_id] = document
        self.tokenized_docs[doc_id] = self.preprocess_text(document)
        
        logger.info(f"Rebuilding BM25 index with {len(self.documents)} total documents")
        # Rebuild BM25 index with all documents
        doc_list = [self.tokenized_docs[id] for id in sorted(self.tokenized_docs.keys())]
        self.bm25 = BM25Okapi(doc_list)
        
        self.modified = True
        
        # Handle persistence
        should_persist = persist if persist is not None else self.auto_persist
        if should_persist:
            self.save_index()
    
    def add_documents_batch(self, doc_dict: Dict[str, str], persist: bool = None) -> None:
        """
        Add multiple documents to the index efficiently
        
        Args:
            doc_dict: Dictionary mapping doc_ids to document content
            persist: Override auto_persist setting for this operation
        """
        logger.info(f"Adding batch of {len(doc_dict)} documents")
        
        # Process all documents first
        for doc_id, document in doc_dict.items():
            logger.debug(f"Processing document {doc_id}: {len(document)} characters")
            self.documents[doc_id] = document
            self.tokenized_docs[doc_id] = self.preprocess_text(document)
        
        # Rebuild index once for all documents
        logger.info(f"Rebuilding BM25 index with {len(self.documents)} total documents")
        doc_list = [self.tokenized_docs[id] for id in sorted(self.tokenized_docs.keys())]
        self.bm25 = BM25Okapi(doc_list)
        
        self.modified = True
        
        # Handle persistence
        should_persist = persist if persist is not None else self.auto_persist
        if should_persist:
            self.save_index()
    
    def add_document_chunks(self, chunks, file_id):
        logger.info(f"Going to add document chunks to BM25 Index for file {file_id}")
        # Use the file_id as a prefix for chunk IDs
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        doc_dict = {}
        i = 0
        for chunk in chunks:
            chunk_id = ids[i]
            doc_dict[chunk_id] = chunk
            i = i + 1
        self.add_documents_batch(doc_dict, persist=True) 

        return

    def save_index(self) -> None:
        """Save the BM25 index and documents to disk"""
        if not self.modified:
            logger.debug("Index unchanged, skipping save operation")
            return
            
        logger.info(f"Saving index to {self.index_path}")
        
        save_dict = {
            'bm25': self.bm25,
            'documents': self.documents,
            'tokenized_docs': self.tokenized_docs,
            'timestamp': time.time()
        }
        
        try:
            # Save to temporary file first
            temp_path = f"{self.index_path}.temp"
            with open(temp_path, 'wb') as f:
                pickle.dump(save_dict, f)
            
            # Atomic rename to ensure consistency
            os.replace(temp_path, self.index_path)
            
            self.modified = False
            logger.info(f"Successfully saved index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(self) -> None:
        """Load the BM25 index and documents from disk"""
        if os.path.exists(self.index_path):
            logger.info(f"Loading existing index from {self.index_path}")
            try:
                with open(self.index_path, 'rb') as f:
                    save_dict = pickle.load(f)
                
                self.bm25 = save_dict['bm25']
                self.documents = save_dict['documents']
                self.tokenized_docs = save_dict['tokenized_docs']
                self.modified = False
                
                logger.info(f"Successfully loaded index with {len(self.documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to load index: {str(e)}")
                raise
        else:
            logger.info("No existing index found, initializing empty state")
            self.bm25 = None
            self.documents = {}
            self.tokenized_docs = {}
            self.modified = False

    def search(self, 
               query: str, 
               top_k: int = 5
               ) -> List[Tuple[str, str, float]]:
        """
        Search documents using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, document, score) tuples
        """
        logger.info(f"Processing search query: '{query}'")
        logger.debug(f"Requesting top {top_k} results")
        
        if not self.bm25:
            logger.warning("No documents in index, returning empty results")
            return []
        
        # Tokenize query
        tokenized_query = self.preprocess_text(query)
        logger.debug(f"Tokenized query into {len(tokenized_query)} tokens")
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create document-score pairs with doc_ids
        doc_scores = []
        for (doc_id, _), score in zip(sorted(self.documents.items()), scores):
            doc_scores.append((doc_id, self.documents[doc_id], score))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[2], reverse=True)
        
        top_results = doc_scores[:top_k]
        logger.info(f"Found {len(top_results)} results")
        logger.debug(f"Top score: {top_results[0][2] if top_results else 'N/A'}")
        
        return top_results
