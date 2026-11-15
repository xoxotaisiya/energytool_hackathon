"""Необязательный реранкер платформы; Ollama не предоставляет /rerank."""
import os
import httpx


class ApiReranker:
    def __init__(self):
        self.base = (os.getenv("RERANKER_BASE_URL") or os.getenv("LLM_BASE_URL")
                     or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "").rstrip("/")
        if not self.base:
            raise ValueError("Не задан адрес API реранкера")
        self.key = os.getenv("RERANKER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("RERANKER_MODEL", "qwen/qwen3-reranker-8b")
        self.client = httpx.Client(timeout=30)

    def rerank(self, query, candidates, top_n):
        texts = [c["path"] + "\n" + c["text"] for c in candidates]
        response = self.client.post(self.base + "/rerank",
                                    headers={"Authorization": "Bearer " + self.key},
                                    json={"model": self.model, "query": query, "documents": texts, "top_n": top_n})
        response.raise_for_status()
        data = response.json()
        rows = data.get("results", data.get("data", []))
        indices = [row["index"] for row in rows]
        if (not indices or any(type(i) is not int or not 0 <= i < len(candidates) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Некорректные индексы реранкера")
        return indices[:top_n]

    def close(self):
        self.client.close()
