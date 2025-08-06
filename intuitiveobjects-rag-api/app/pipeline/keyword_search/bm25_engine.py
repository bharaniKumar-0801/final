


import pickle
from rank_bm25 import BM25Okapi
from typing import List


class BM25:
    def __init__(self, tokenized_docs: List[List[str]]):
        self.tokenized_docs = tokenized_docs
        self.bm25 = BM25Okapi(tokenized_docs)

    def get_top_n(self, query_tokens: List[str], n=5):
        scores = self.bm25.get_scores(query_tokens)
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        return top_n

    def save_to_disk(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump({"bm25": self}, f)


    @staticmethod
    def load_from_disk(file_path: str):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            obj = BM25(data["tokenized_docs"]) 
            obj.bm25 = data["bm25"] 
            return obj
