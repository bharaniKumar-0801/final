import os
import pickle
import json
from app.pipeline.keyword_search.bm25_engine import BM25
from app.pipeline.keyword_search.text_utils import get_keywords

BM25_STORE = "app/pipeline/bm25_store"

def query_all_bm25(query: str, top_n: int = 5):
    query_tokens = get_keywords(query)
    all_results = []

    for file in os.listdir(BM25_STORE):
        if file.endswith(".pkl") :
            file_id = file.replace(".pkl", "")
            try:
                bm25 = BM25.load_from_disk(os.path.join(BM25_STORE, file))
                # with open(os.path.join(BM25_STORE, f"{file_id}_metadata.pkl"), "rb") as f:
                #     metadata = pickle.load(f)
                with open(os.path.join(BM25_STORE, f"{file_id}_texts.json")) as f:
                    texts = json.load(f)

                top_chunks = bm25.get_top_n(query_tokens, top_n)
                for idx, score in top_chunks:
                    all_results.append({
                        "score": score,
                        "text": texts[idx],
                        # "metadata": metadata[idx],
                        "file_id": file_id
                    })

            except Exception as e:
                print(f"⚠️ Failed to load BM25 index for {file_id}: {e}")

    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:top_n]
    return sorted_results
