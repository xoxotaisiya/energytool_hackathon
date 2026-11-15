"""Подготовка кеша BGE; не является измерением качества поиска."""
import json
import time
from pathlib import Path
import httpx
import config
from rag.retrieval import Retriever


def main():
    deadline = time.monotonic() + 7200
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                data = client.get(config.OLLAMA_BASE_URL + "/api/tags").json()
                if any(m["name"] in {"bge-m3", "bge-m3:latest"} for m in data.get("models", [])):
                    break
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            time.sleep(10)
        else:
            raise RuntimeError("BGE не загружена")
    started = time.perf_counter()
    r = Retriever(mode="ollama")
    try:
        if r.mode != "ollama":
            raise RuntimeError("Embeddings не запущены")
        result = {"mode": r.mode, "chunks": len(r.chunks), "seconds": round(time.perf_counter() - started, 2)}
        path = config.ROOT / "reports/embedding_cache.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    finally:
        r.close()


if __name__ == "__main__":
    main()
