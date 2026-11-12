"""Эмбеддинги текста: два взаимозаменяемых источника.

LocalEmbedder — bge-m3 запускается на вашем компьютере (для разработки).
ApiEmbedder   — тот же bge-m3 через прокси платформы (POST /embeddings), в совместимом API.

Оба возвращают numpy-матрицу с нормированными векторами, поэтому скалярное
произведение двух векторов = косинусная близость.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _normalize(m):
    """Нормализует векторы, чтобы сравнивать их по косинусной близости."""
    m = np.asarray(m, dtype="float32")
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


class LocalEmbedder:
    def __init__(self, model="BAAI/bge-m3"):
        """Загружает модель эмбеддингов или настраивает API-клиент."""
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model, device="cpu")

    def encode(self, texts, batch_size=8):
        """Преобразует список текстов в векторы."""
        return _normalize(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False))


class ApiEmbedder:
    """Использует переменные окружения платформы: LLM_BASE_URL, LLM_API_KEY, EMB_MODEL."""

    def __init__(self):
        """Загружает модель эмбеддингов или настраивает API-клиент."""
        import httpx
        self.base = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.key = os.getenv("EMBEDDER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("EMB_MODEL", "baai/bge-m3")
        self.client = httpx.Client(timeout=90)

    def encode(self, texts, batch_size=32):
        """Преобразует список текстов в векторы."""
        out = []
        for i in range(0, len(texts), batch_size):
            body = {"model": self.model_name, "input": texts[i:i + batch_size]}
            for attempt in range(3):
                try:
                    r = self.client.post(self.base + "/embeddings",
                                         headers={"Authorization": "Bearer " + self.key}, json=body)
                    r.raise_for_status()
                    data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
                    out += [d["embedding"] for d in data]
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
        return _normalize(out)


def chunk_texts(chunks):
    """Что именно превращается в вектор: путь по заголовкам + текст фрагмента."""
    return [c["path"] + "\n" + c["text"] for c in chunks]


def embed_chunks_cached(embedder, chunks):
    """Для разработки: кэширует векторы на диск, чтобы не считать заново при каждом запуске."""
    texts = chunk_texts(chunks)
    h = hashlib.sha1((embedder.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"emb_{h}.npy"
    if f.exists():
        return np.load(f)
    m = embedder.encode(texts)
    np.save(f, m)
    return m
