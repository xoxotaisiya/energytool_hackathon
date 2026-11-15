"""Поиск по базе знаний Energytool: единая точка входа для остальных модулей бота.

Как пользоваться (например, из solution.py):

    from rag.retrieval import Retriever

    retriever = Retriever()                       # один раз при запуске: читает базу, строит индекс
    result = retriever.retrieve(question)         # question может содержать диалог целиком
    result = retriever.retrieve(question, history=[{"role": "user", "content": "..."},
                                                   {"role": "assistant", "content": "..."}])
    result["chunks"]        # найденные фрагменты, лучший первым
    result["no_answer"]     # True — в базе нет подходящего
    result["confidence"]    # сходство лучшего фрагмента (0..1), None в запасном режиме
    result["guards"]        # защитные блоки найденного продукта («Чего утверждать нельзя» и т.п.)

Отладка: RETRIEVER_DEBUG=1 — на каждый вопрос строка в stderr (режим, уверенность, id фрагментов).

Режимы поиска (см. result["mode"]):
    "dense"  — смысловой поиск bge-m3 (через прокси платформы, переменные окружения LLM_BASE_URL,
               LLM_API_KEY, EMB_MODEL), основной;
    "local"  — тот же bge-m3 на своём компьютере (только для разработки, нужен sentence-transformers);
    "bm25"   — запасной словарный поиск без внешних вызовов: включается сам, если прокси недоступен.

Модуль не запускает модель-генератор, не хранит диалоги и не пишет ответы: он только ищет.
Модуль находится в пакете rag и используется через from rag.retrieval import Retriever.
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from rag.build_chunks import load_docs, make_chunks
from rag.retriever import BM25Index
from rag.fusion import rrf
from rag.query_normalization import normalize_query, infer_products, dialog_products, question_purpose, bill_problem, usage_comparison

HERE = Path(__file__).resolve().parent

TOP_K = 5                  # сколько фрагментов отдавать
THRESHOLD = 0.50           # сходство лучшего фрагмента ниже этого значения -> no_answer
QUERY_MODE = "mixa"        # как строить запрос из диалога (см. build_queries): две оценки складываются

SPEAKERS = {"пользователь": "user", "клиент": "user", "user": "user",
            "ассистент": "assistant", "бот": "assistant", "assistant": "assistant"}
SPEAKER_RE = re.compile(r"(?mi)^\s*(Пользователь|Клиент|Ассистент|Бот|User|Assistant)\s*:\s*")


def _env(name, default=""):
    """Читает переменную окружения или возвращает значение по умолчанию."""
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def log(msg):
    """Пишет диагностическое сообщение в stderr."""
    print(f"[retrieval] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------ диалог -> запрос
def parse_dialog(text):
    """'Пользователь: ...\\nАссистент: ...' -> [('user', '...'), ('assistant', '...')].
    Если пометок нет, весь текст считается одной репликой пользователя."""
    parts = SPEAKER_RE.split(text)
    if len(parts) < 3:
        return [("user", text.strip())]
    turns = []
    for who, said in zip(parts[1::2], parts[2::2]):
        said = said.strip()
        if said:
            turns.append((SPEAKERS[who.lower()], said))
    return turns or [("user", "")]


def normalize_history(history):
    """[{'role','content'}, ('user','текст'), ...] -> [('user','текст'), ...]."""
    turns = []
    for item in history or []:
        if isinstance(item, dict):
            role, said = item.get("role", "user"), item.get("content") or item.get("text") or ""
        else:
            role, said = item
        role = SPEAKERS.get(str(role).lower(), "assistant" if str(role).lower() != "user" else "user")
        if str(said).strip():
            turns.append((role, str(said).strip()))
    return turns


def build_query(turns, mode=QUERY_MODE):
    """Из реплик собирает строку для поиска.
    last  — только последняя реплика клиента;
    last2 — две последние реплики клиента;
    users — все реплики клиента (ответы ассистента отбрасываются);
    full  — весь диалог как есть."""
    users = [t for r, t in turns if r == "user"] or [turns[-1][1]]
    if mode in ("mix", "mixa"):    # для текстового вида (например, BM25): вся речь клиента
        return " ".join(users)
    if mode == "last":
        return users[-1]
    if mode == "last2":
        return " ".join(users[-2:])
    if mode == "full":
        return "\n".join(("Пользователь: " if r == "user" else "Ассистент: ") + t for r, t in turns)
    return " ".join(users)


def build_queries(turns, mode=QUERY_MODE, w_last=0.5):
    """Список (текст запроса, вес). В режиме 'mix' ищем по двум запросам сразу: по последней реплике
    клиента (о чём он спрашивает сейчас) и по всей его речи (о ком и о чём разговор), оценки складываются."""
    if mode in ("mix", "mixa"):
        users = [t for r, t in turns if r == "user"] or [turns[-1][1]]
        if len(users) > 1:
            context = " ".join(users)
            if mode == "mixa":             # плюс последняя реплика ассистента: тему мог назвать он
                assistants = [t for r, t in turns if r == "assistant"]
                if assistants:
                    context = assistants[-1] + " " + context
            return [(users[-1], w_last), (context, 1 - w_last)]
    return [(build_query(turns, mode), 1.0)]


# ------------------------------------------------------------ база знаний
def find_kb():
    """Путь к базе: KB_PATH, затем DATA_DIR/knowledge-base.zip (как на платформе), затем ../knowledge-base."""
    if _env("KB_PATH"):
        return Path(_env("KB_PATH"))
    data = Path(_env("DATA_DIR", "data"))
    for root in (data, HERE / data, Path.cwd(), HERE):
        for name in ("knowledge-base.zip", "knowledge_base.zip", "kb.zip"):
            if (root / name).is_file():
                return root / name
    if (HERE.parent / "knowledge-base").is_dir():
        return HERE.parent / "knowledge-base"
    raise FileNotFoundError("Не найдена база знаний: положите knowledge-base.zip в DATA_DIR или задайте KB_PATH")


# ------------------------------------------------------------ поиск
class Retriever:
    def __init__(self, kb_path=None, mode=None, top_k=None, threshold=None, query_mode=None):
        """Загружает базу, строит индекс и выбирает режим поиска."""
        t0 = time.time()
        self.top_k = top_k or int(_env("TOP_K", TOP_K))
        self.threshold = float(_env("RETRIEVER_THRESHOLD", THRESHOLD)) if threshold is None else threshold
        self.query_mode = query_mode or _env("QUERY_MODE", QUERY_MODE)
        self.normalize_queries = _env("QUERY_NORMALIZATION", "1").lower() not in {"0", "false", "no", "off"}
        self.w_last = float(_env("W_LAST", 0.5))       # вес последней реплики в режиме 'mix'
        self.candidate_k = int(_env("CANDIDATE_K", 30))
        self.rrf_k = int(_env("RRF_K", 60))
        self.dense_weight = float(_env("DENSE_WEIGHT", 0.6))
        self.bm25_weight = float(_env("BM25_WEIGHT", 0.4))
        if self.top_k <= 0 or self.candidate_k <= 0 or self.rrf_k <= 0:
            raise ValueError("TOP_K, CANDIDATE_K и RRF_K должны быть положительными")
        if not 0 <= self.w_last <= 1 or self.dense_weight < 0 or self.bm25_weight < 0 or not (self.dense_weight + self.bm25_weight):
            raise ValueError("Некорректные веса поиска")
        self.debug = _env("RETRIEVER_DEBUG").lower() in ("1", "true", "yes", "on")
        self.with_guards = _env("RETRIEVER_GUARDS", "1").lower() not in ("0", "false", "no", "off")
        self.reranker = None
        self.reranker_requested = _env("RERANKER_ENABLED", "0").lower() in ("1", "true", "yes", "on")
        if self.reranker_requested:
            try:
                if _env("RERANKER_BACKEND", "api") == "ollama":
                    from rag.ollama_reranker import OllamaReranker
                    self.reranker = OllamaReranker()
                else:
                    from rag.reranker import ApiReranker
                    self.reranker = ApiReranker()
            except Exception as exc:
                log(f"Реранкер недоступен ({type(exc).__name__}); используется исходное ранжирование")

        docs = load_docs(kb_path or find_kb())
        self.chunks = make_chunks(docs)[0]
        self.bm25 = BM25Index(self.chunks, 5, 3)
        # защитные блоки по продуктам: сначала «нельзя утверждать», потом «не делаем», границы, «не подходит»
        order = {"forbidden": 0, "we-dont": 1, "limits": 2, "not-for": 3}
        self.guards_by_product = {}
        for i, c in sorted(enumerate(self.chunks), key=lambda x: order.get(x[1]["section_type"], 9)):
            if c["guard"]:
                self.guards_by_product.setdefault(c["product"], []).append(i)

        want = (mode or _env("RETRIEVER_MODE") or "auto").lower()
        if want == "auto":
            want = "dense" if _env("LLM_BASE_URL") or _env("OPENAI_BASE_URL") or _env("BASE_URL") else "bm25"
        if want not in ("dense", "local", "bm25", "hybrid", "hybrid-local", "ollama", "hybrid-ollama"):
            raise ValueError("Неизвестный RETRIEVER_MODE: " + want)
        self.mode, self.embedder, self.matrix, self._qcache = "bm25", None, None, {}
        if want != "bm25":
            try:
                backend = {"hybrid": "dense", "hybrid-local": "local", "hybrid-ollama": "ollama"}.get(want, want)
                self.embedder = self._make_embedder(backend)
                self.matrix = self._embed_chunks(backend)
                if len(self.matrix) != len(self.chunks):
                    raise ValueError("Число векторов не совпадает с числом фрагментов")
                self.mode = want
            except Exception as exc:
                log(f"Эмбеддинги недоступны ({type(exc).__name__}); используется BM25")
                self.embedder = None
        self.startup_s = time.time() - t0
        log(f"режим={self.mode}; фрагментов={len(self.chunks)}; запуск {self.startup_s:.1f} с")

    @staticmethod
    def _make_embedder(mode):
        """Создает локальный или API-клиент эмбеддингов."""
        if mode == "local":
            from rag.embedder import LocalEmbedder
            return LocalEmbedder()
        if mode == "ollama":
            from rag.embedder import OllamaEmbedder
            return OllamaEmbedder()
        from rag.embedder import ApiEmbedder
        return ApiEmbedder()

    def _embed_chunks(self, mode):
        """Строит векторы для всех фрагментов базы."""
        from rag.embedder import chunk_texts, embed_chunks_cached
        if mode in ("local", "ollama"):
            return embed_chunks_cached(self.embedder, self.chunks)     # кэш на диске, для разработки
        return self.embedder.encode(chunk_texts(self.chunks))

    def _dense(self, queries, k):
        """Ищет фрагменты по смысловой близости векторов."""
        sims = 0
        for text, weight in queries:
            if text not in self._qcache:
                self._qcache[text] = self.embedder.encode([text])[0]
            sims = sims + weight * (self.matrix @ self._qcache[text])
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def prefetch(self, questions):
        """Считает embeddings набора вопросов пакетно до загрузки генератора."""
        if self.embedder is None:
            return 0
        texts = list(dict.fromkeys(text for question in questions
                                  for text, _ in self._queries(parse_dialog(question))
                                  if text not in self._qcache))
        if texts:
            vectors = self.embedder.encode(texts)
            if len(vectors) != len(texts):
                raise ValueError("Неполные embeddings запросов")
            self._qcache.update(zip(texts, vectors))
        return len(texts)

    def _queries(self, turns):
        queries = build_queries(turns, self.query_mode, self.w_last)
        return [(normalize_query(text), weight) for text, weight in queries] if self.normalize_queries else queries

    def retrieve(self, question, history=None, top_k=None):
        """Ищет релевантные фрагменты по вопросу и истории диалога."""
        t0 = time.time()
        k = top_k or self.top_k
        if k <= 0:
            raise ValueError("top_k должен быть положительным")
        turns = normalize_history(history) + parse_dialog(question)
        if not any(t for _, t in turns):                      # пустой вопрос: искать нечего
            return {"query": "", "mode": self.mode, "confidence": None, "no_answer": True,
                    "search_ms": 0, "chunks": [], "guards": []}
        query = build_query(turns, self.query_mode)           # для BM25 и для показа
        original_query = query
        if self.normalize_queries:
            query = normalize_query(query)
        queries = self._queries(turns)
        shown = " ‖ ".join(t for t, _ in queries) if len(queries) > 1 else query

        final_k = k
        if self.reranker is not None:
            k = max(k, self.candidate_k)
        mode, hits, confidence = self.mode, None, None
        if self.embedder is not None:
            try:
                dense_hits = self._dense(queries, max(k, self.candidate_k) if mode.startswith("hybrid") else k)
                confidence = round(dense_hits[0][1], 3) if dense_hits else None
                hits = rrf([dense_hits, self.bm25.search(query, max(k, self.candidate_k))],
                           [self.dense_weight, self.bm25_weight], self.rrf_k)[:k] if mode.startswith("hybrid") else dense_hits
            except Exception as exc:                       # разовый сбой прокси -> BM25 для этого вопроса
                log(f"эмбеддинг запроса не получен ({type(exc).__name__}); BM25 для этого вопроса")
                mode = "bm25"
        else:
            mode = "bm25"
        if hits is None:
            hits = self.bm25.search(query, k)

        if mode == "bm25":
            confidence, no_answer = None, not hits
        else:
            no_answer = confidence is None or confidence < self.threshold
        reranked, rerank_error = False, None
        if self.reranker is not None and hits:
            try:
                indices = self.reranker.rerank(query, [self.chunks[i] for i, _ in hits], final_k)
                # Некоторые сервисы возвращают неполный список. Реранжирование
                # меняет порядок, но не должно терять остальные кандидаты.
                indices = list(indices) + [i for i in range(len(hits)) if i not in indices]
                hits = [hits[i] for i in indices]
                reranked = True
            except Exception as exc:
                rerank_error = type(exc).__name__
        hits = hits[:final_k]
        result = {
            "query": shown,
            "original_query": original_query,
            "mode": mode,
            "confidence": confidence,
            "confidence_kind": "dense_cosine" if mode != "bm25" else None,
            "score_kind": "rrf" if mode.startswith("hybrid") else ("bm25" if mode == "bm25" else "cosine"),
            "reranked": reranked,
            "rerank_error": rerank_error or ("unavailable" if self.reranker_requested and self.reranker is None else None),
            "no_answer": no_answer,
            "search_ms": round((time.time() - t0) * 1000),
            "chunks": [self._chunk(i, s) for i, s in hits],
            "supporting_chunks": self._supporting(turns, hits),
            "guards": self._guards(hits, query),
        }
        if self.debug:
            log(f"режим={mode} уверенность={confidence} no_answer={no_answer} {result['search_ms']} мс "
                f"фрагменты={[c['id'] for c in result['chunks']]} защитные={[c['id'] for c in result['guards']]} "
                f"запрос={query[-70:]!r}")
        return result

    def _supporting(self, turns, hits):
        latest = next((text for role, text in reversed(turns) if role == "user"), "")
        products = dialog_products([{"role": role, "content": text} for role, text in turns])
        shown = {i for i, _ in hits}
        # Структурное расширение по продукту и типу раздела: перечень необходимых
        # данных берётся из самой базы, не из зашитого списка ответов.
        indices = []
        if bill_problem(latest):
            indices += [i for i, c in enumerate(self.chunks) if i not in shown
                        and re.search(r"почему раст[её]т сч[её]т", c["path"], re.I)
                        and c["section_type"] not in {"forbidden", "we-dont"}][:2]
        if re.search(r"цех", latest, re.I) or usage_comparison(latest):
            indices += [i for i, c in enumerate(self.chunks) if i not in shown and c["product"] == "askue"
                        and "цех" in c["text"].lower()][:2]
        if "orem" in products and re.search(r"мощност|подход|доступ|участ|выйти|выход|\d\s*(?:кВт|МВт)", latest, re.I):
            indices += [i for i, c in enumerate(self.chunks) if i not in shown and c["product"] == "orem"
                        and re.search(r"менее\s+\d+\s+МВт|от\s+\d+\s+МВт.*прям", c["text"], re.I)][:3]
        if "solar" in products and re.search(r"окуп|экономическ.*смысл", latest, re.I):
            indices += [i for i, c in enumerate(self.chunks) if i not in shown and c["product"] == "solar"
                        and re.search(r"в расч[её]те учитываются регион|разброс определяется регионом", c["text"], re.I)][:2]
        advice = re.search(r"^(?:мы\s+)?(?:хотим|планируем|нам нужна?|нужна?)\b", latest, re.I) and "?" not in latest
        for product in products if question_purpose(latest) == "inputs" or advice else []:
            matching = [i for i, c in enumerate(self.chunks) if c["product"] == product
                        and c["section_type"] == "inputs-needed" and i not in shown]
            indices.extend(matching[:2])
        return [self._chunk(i, None) for i in dict.fromkeys(indices)]

    def _guards(self, hits, query=""):
        """Защитные блоки продукта, который нашёлся первым и которых ещё нет среди найденных фрагментов:
        «Чего утверждать нельзя», «Что мы не делаем», «Не подходит, если», границы продукта."""
        if not hits or not self.with_guards:
            return []
        shown = {i for i, _ in hits}
        products = infer_products(query) or list(dict.fromkeys(self.chunks[i]["product"] for i, _ in hits[:3]))[:2]
        products += ["general"]
        indices = dict.fromkeys(i for product in products for i in self.guards_by_product.get(product, []))
        return [self._chunk(i, None) for i in indices if i not in shown]

    def _chunk(self, i, score):
        """Возвращает фрагмент с метаданными и оценкой."""
        c = self.chunks[i]
        return {"id": c["id"], "file": c["file"], "topic": c["topic"],
                "product": c["product"], "path": c["path"],
                "section_type": c["section_type"],
                "score": None if score is None else round(score, 3),
                "guard": c["guard"], "text": c["text"]}

    def close(self):
        client = getattr(self.embedder, "client", None)
        if client is not None:
            client.close()
        if self.reranker is not None:
            self.reranker.close()


# ------------------------------------------------------------ запуск из командной строки
if __name__ == "__main__":
    import argparse
    import json
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Поиск по базе знаний Energytool")
    ap.add_argument("question", help="вопрос (можно целиком диалог в формате 'Пользователь: ...')")
    ap.add_argument("--mode", choices=["auto", "dense", "local", "bm25", "hybrid", "hybrid-local", "ollama", "hybrid-ollama"], help="режим поиска")
    ap.add_argument("--short", action="store_true", help="тексты фрагментов сократить")
    args = ap.parse_args()
    out = Retriever(mode=args.mode).retrieve(args.question)
    if args.short:
        for c in out["chunks"]:
            c["text"] = c["text"][:100] + "…"
    print(json.dumps(out, ensure_ascii=False, indent=2))
