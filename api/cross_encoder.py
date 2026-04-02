import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


_DEFAULT_MAX_WORKERS = int(os.getenv("RERANK_MAX_WORKERS", "6"))
_DEFAULT_MAX_IN_FLIGHT = int(os.getenv("RERANK_MAX_IN_FLIGHT", "12"))

# Reuse a single bounded pool per process so multiple concurrent requests
# don't create unbounded threads and overwhelm the host / HF API.
_executor = ThreadPoolExecutor(max_workers=_DEFAULT_MAX_WORKERS)
_in_flight = threading.Semaphore(_DEFAULT_MAX_IN_FLIGHT)


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_workers=6):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.api_token = os.getenv("HF_TOKEN")
        self.max_workers = max_workers
        if not self.api_token:
            raise ValueError("Set HF_TOKEN in your environment for Hugging Face Inference API access.")

    def _score_one(self, query, doc):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "")
        payload = {"inputs": {"source_sentence": query, "sentences": [content]}}
        try:
            if not _in_flight.acquire(timeout=15):
                return 0
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
            finally:
                _in_flight.release()
            if response.status_code == 200:
                result = response.json()
                return result[0] if isinstance(result, list) else result.get("score", 0)
        except Exception:
            pass
        return 0

    def rerank(self, query, docs, top_k=6):
        if not docs:
            return []
        scores = []
        # Use shared executor; also cap work submitted per call.
        to_score = docs[: min(len(docs), max(1, int(self.max_workers)))]
        future_to_doc = {_executor.submit(self._score_one, query, doc): doc for doc in to_score}
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                score = future.result()
            except Exception:
                score = 0
            scores.append((doc, score))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
