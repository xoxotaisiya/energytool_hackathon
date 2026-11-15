"""Шаблон точки входа платформы. tools.build_submission переносит его в solution.py."""
import csv
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def value(*names, default=""):
    return next((os.environ[n].strip() for n in names if os.environ.get(n, "").strip()
                 and os.environ[n].strip() != n), default)


def platform_environment():
    if not value("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"):
        raise ValueError("Не передан адрес API платформы")
    if not value("LLM_API_KEY", "OPENAI_API_KEY"):
        raise ValueError("Не передан ключ API платформы")
    if not value("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL"):
        raise ValueError("Не передан идентификатор генеративной модели платформы")
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["RERANKER_BACKEND"] = "api"
    mode = value("RETRIEVER_MODE", default="hybrid")
    if mode not in {"auto", "dense", "hybrid", "bm25"}:
        raise ValueError("Для платформы нужен поиск через её API либо BM25")
    os.environ["RETRIEVER_MODE"] = mode
    os.environ.setdefault("TOP_K", "8")


def find_file(root, *names):
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError("Не найден обязательный входной файл: " + names[0])


def read_questions(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "question"}.issubset(reader.fieldnames or []):
            raise ValueError("В questions.csv нужны id и question")
        rows = list(reader)
    ids = [r["id"] for r in rows]
    if any(not qid for qid in ids) or len(ids) != len(set(ids)):
        raise ValueError("ID должны быть непустыми и уникальными")
    if any(not isinstance(r.get("question"), str) or not r["question"].strip() for r in rows):
        raise ValueError("Пустой тестовый вопрос")
    return rows


def write_answers(path, rows, answers):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        writer.writerows((row["id"], answer) for row, answer in zip(rows, answers))


def main():
    platform_environment()
    # Импорты выполняются после выбора API-профиля; Ollama не запускается.
    import config
    from consultant.consultant import EnergytoolConsultant
    from memory.conversation_state import ConversationState
    from rag.retrieval import Retriever, parse_dialog

    root = Path(value("DATA_DIR", default="data"))
    rows = read_questions(find_file(root, "questions.csv"))
    kb = find_file(root, "knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    workers = int(value("MAX_WORKERS", default="3"))
    if not 1 <= workers <= 8:
        raise ValueError("MAX_WORKERS должен быть в диапазоне 1..8")
    failure = "Сейчас не удалось получить проверенный ответ модели. Повторите запрос."
    answers = [failure for _ in rows]
    bot = None
    try:
        with tempfile.TemporaryDirectory(prefix="energytool-platform-") as session_dir:
            config.SESSION_DIR = session_dir
            bot = EnergytoolConsultant(retriever=Retriever(kb_path=kb, mode=config.RETRIEVER_MODE))
            bot.llm.check_connection()
            try:
                bot.retriever.prefetch([row["question"] for row in rows])
            except Exception:
                # Отказ embeddings не должен блокировать генератор, если
                # его API доступен. Индекс BM25 всегда уже построен.
                print("prefetch_fallback=BM25", file=sys.stderr)
                client = getattr(bot.retriever.embedder, "client", None)
                if client is not None:
                    client.close()
                bot.retriever.embedder = bot.retriever.matrix = None
                bot.retriever.mode = "bm25"
            def answer_one(row):
                turns = parse_dialog(row["question"])
                state = ConversationState()
                for role, text in turns[:-1]:
                    state.add(role, text)
                result = bot.reply(turns[-1][1], state)
                answer = result["answer"].strip()
                if not answer or len(answer) > 10000:
                    raise ValueError("Некорректная длина ответа")
                return answer, result["debug"].get("error_code")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(answer_one, row): i for i, row in enumerate(rows)}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        answers[index], error = future.result()
                        if error:
                            print(f"case_index={index} error_code={error}", file=sys.stderr)
                    except Exception as exc:
                        print(f"case_index={index} error_type={type(exc).__name__}", file=sys.stderr)
    finally:
        # Даже технический сбой не подменяется выдуманными продуктовым ответом.
        # Файл сохраняет исходные ID, порядок и стандартное CSV-экранирование.
        write_answers(Path("submission.csv"), rows, answers)
        if bot is not None:
            bot.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("platform_error_type=" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(2)
