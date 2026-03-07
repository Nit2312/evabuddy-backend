import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


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
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
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
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(docs))) as executor:
            future_to_doc = {executor.submit(self._score_one, query, doc): doc for doc in docs}
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    score = future.result()
                except Exception:
                    score = 0
                scores.append((doc, score))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
