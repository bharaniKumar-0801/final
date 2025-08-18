#from .bm25_impl import BM25Searcher
import textwrap

from rank_bm25 import BM25Okapi
from typing import List, Tuple
import pickle
from pathlib import Path

class BM25Searcher:
    def __init__(self):
        """
        Initialize the BM25 searcher
        """
        self.bm25 = None
        self.documents = None
        self.tokenized_docs = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess and tokenize text
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Basic preprocessing - you can enhance this based on your needs
        return text.lower().split()
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents using BM25
        
        Args:
            documents: List of text documents/chunks
        """
        self.documents = documents
        # Tokenize all documents
        self.tokenized_docs = [self.preprocess_text(doc) for doc in documents]
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25:
            raise ValueError("No documents have been indexed yet")
        
        # Tokenize query
        tokenized_query = self.preprocess_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create document-score pairs
        doc_scores = list(zip(self.documents, scores))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def save_index(self, path: str) -> None:
        """
        Save the BM25 index and documents to disk
        
        Args:
            path: Path to save the index
        """
        if not self.bm25:
            raise ValueError("No documents have been indexed yet")
        
        save_dict = {
            'bm25': self.bm25,
            'documents': self.documents,
            'tokenized_docs': self.tokenized_docs
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load_index(self, path: str) -> None:
        """
        Load the BM25 index and documents from disk
        
        Args:
            path: Path to load the index from
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.bm25 = save_dict['bm25']
        self.documents = save_dict['documents']
        self.tokenized_docs = save_dict['tokenized_docs']

def main():
    # Create some sample documents
    documents = [
        """Python is a high-level programming language known for its simplicity 
        and readability. It supports multiple programming paradigms, including 
        procedural, object-oriented, and functional programming.""",
        
        """Machine learning is a subset of artificial intelligence that enables 
        systems to learn and improve from experience without being explicitly 
        programmed.""",
        
        """Deep learning is part of machine learning based on artificial neural 
        networks. It allows computational models to learn data representations 
        with multiple levels of abstraction.""",
        
        """Natural Language Processing (NLP) is a branch of artificial intelligence 
        that helps computers understand, interpret, and manipulate human language.""",
        
        """The Python programming language is widely used in data science and 
        machine learning applications due to its rich ecosystem of libraries."""
    ]
    
    # Clean up the documents by removing extra whitespace
    documents = [' '.join(doc.split()) for doc in documents]
    
    # Initialize BM25 searcher
    print("Initializing BM25 searcher...")
    bm25_searcher = BM25Searcher()
    
    # Index the documents
    print("Indexing documents...\n")
    bm25_searcher.index_documents(documents)
    
    # Test queries
    test_queries = [
        "What is Python programming?",
        "machine learning and artificial intelligence",
        "neural networks deep learning",
        "NLP human language",
    ]
    
    # Run searches
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = bm25_searcher.search(query, top_k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nRank {i} (Score: {score:.4f}):")
            print(textwrap.fill(doc, width=75))
    
    # Test save and load functionality
    print("\n\nTesting save/load functionality...")
    
    # Save the index
    bm25_searcher.save_index("test_bm25_index.pkl")
    
    # Create a new searcher and load the index
    new_searcher = BM25Searcher()
    new_searcher.load_index("test_bm25_index.pkl")
    
    # Verify the loaded index works
    print("\nTesting loaded index:")
    results = new_searcher.search("Python programming", top_k=1)
    print(f"\nTop result (Score: {results[0][1]:.4f}):")
    print(textwrap.fill(results[0][0], width=75))

if __name__ == "__main__":
    main()
