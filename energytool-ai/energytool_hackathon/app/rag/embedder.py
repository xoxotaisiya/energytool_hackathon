"""Эмбеддинги текста: два взаимозаменяемых источника.

LocalEmbedder — bge-m3 запускается на вашем компьютере (для разработки).
ApiEmbedder   — тот же bge-m3 через прокси платформы (POST /embeddings), в совместимом API.

Оба возвращают numpy-матрицу с нормированными векторами, поэтому скалярное
произведение двух векторов = косинусная близость.
"""
import hashlib
import os
import time
import tempfile
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
        self.base = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "").strip().rstrip("/")
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
                    data = sorted(r.json()["data"], key=lambda d: d["index"])
                    count = len(texts[i:i + batch_size])
                    if len(data) != count or [d["index"] for d in data] != list(range(count)):
                        raise ValueError("Неполный или повреждённый набор embeddings")
                    out += [d["embedding"] for d in data]
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
        matrix = np.asarray(out, dtype="float32")
        if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Недопустимые размеры или значения embeddings")
        if (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError("Получен нулевой embedding")
        return _normalize(matrix)


def chunk_texts(chunks):
    """Что именно превращается в вектор: путь по заголовкам + текст фрагмента."""
    return [c["path"] + "\n" + c["text"] for c in chunks]


class OllamaEmbedder:
    """Локальные embeddings bge-m3 через Ollama, без sentence-transformers."""
    def __init__(self):
        import httpx
        self.base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model_name = os.getenv("OLLAMA_EMB_MODEL", "bge-m3")
        self.client = httpx.Client(timeout=180)

    def encode(self, texts, batch_size=8):
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self.client.post(self.base + "/api/embed",
                                        json={"model": self.model_name, "input": batch, "truncate": False,
                                              "options": {"num_gpu": int(os.getenv("OLLAMA_EMBED_GPU_LAYERS", "0"))}})
            response.raise_for_status()
            rows = response.json()["embeddings"]
            if len(rows) != len(batch):
                raise ValueError("Неполные embeddings Ollama")
            vectors.extend(rows)
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Некорректные embeddings Ollama")
        if (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError("Нулевой embedding Ollama")
        return _normalize(matrix)


def embed_chunks_cached(embedder, chunks):
    """Для разработки: кэширует векторы на диск, чтобы не считать заново при каждом запуске."""
    texts = chunk_texts(chunks)
    h = hashlib.sha1((embedder.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"emb_{h}.npy"
    if f.exists():
        cached = np.load(f)
        if cached.ndim == 2 and len(cached) == len(chunks) and np.isfinite(cached).all():
            return cached
    m = embedder.encode(texts)
    # Другой процесс не должен прочитать наполовину записанный cache.
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".npy", delete=False) as tmp:
        pending = Path(tmp.name)
        np.save(tmp, m)
    try:
        os.replace(pending, f)
    finally:
        pending.unlink(missing_ok=True)
    return m
