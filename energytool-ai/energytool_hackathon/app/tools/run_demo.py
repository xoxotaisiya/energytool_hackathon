"""Прогон реальной модели на независимых demo-сценариях (без заглушек)."""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import config
from consultant.consultant import EnergytoolConsultant
from integration.interface import EnergytoolAI
from llm.qwen import LLMError
from memory.conversation_state import ConversationState
from rag.retrieval import Retriever, parse_dialog


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=config.ROOT / "demo_questions.csv")
    ap.add_argument("--output", type=Path, default=config.ROOT / "reports/demo_bm25.json")
    ap.add_argument("--mode", default="bm25", choices=["bm25", "dense", "local", "hybrid", "hybrid-local", "ollama", "hybrid-ollama"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="+", help="Прогнать только перечисленные ID")
    ap.add_argument("--max-errors", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="Продолжить совместимый прерванный отчёт")
    ap.add_argument("--label", default="", help="Название измерения, например cold или warm")
    ap.add_argument("--unload-model", action="store_true", help="Выгрузить выбранную Qwen из Ollama перед первым вопросом")
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.LLM_MODEL
    source_paths = [config.ROOT / "config.py"] + [p for directory in ("consultant", "rag", "llm", "memory", "integration", "prompts")
                                                    for p in (config.ROOT / directory).glob("*.py")]
    code_hash = hashlib.sha256(b"".join(str(p.relative_to(config.ROOT)).encode() + b"\0" + p.read_bytes()
                                        for p in sorted(source_paths))).hexdigest()
    report = {"completed": False, "provider": config.LLM_PROVIDER, "model": model, "label": args.label,
              "cold_model_requested": args.unload_model,
              "requested_mode": args.mode, "question_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
              "grounded_answers": config.GROUNDED_ANSWERS,
              "code_sha256": code_hash,
              "configuration": {"top_k": os.getenv("TOP_K", "5"), "context_max_chars": config.CONTEXT_MAX_CHARS,
                                "ids": args.ids,
                                "temperature": config.LLM_TEMPERATURE, "max_tokens": config.MAX_TOKENS,
                                "ollama_think": config.OLLAMA_THINK,
                                "query_normalization": os.getenv("QUERY_NORMALIZATION", "1")},
              "results": [], "manual_quality_review_required": True}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("provider", "model", "requested_mode", "question_sha256", "grounded_answers", "code_sha256", "configuration"):
            if previous.get(key) != report[key]:
                ap.error("Нельзя продолжить отчёт с другими параметрами: " + key)
        report["results"] = previous["results"]
    bot = None
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        bot = EnergytoolConsultant(retriever=Retriever(kb_path=config.KB_PATH or None, mode=args.mode))
        if bot.retriever.mode != args.mode:
            raise LLMError("Запрошенный поиск не запущен; fallback не выдаётся за результат сравнения.", "retrieval_fallback")
        with args.questions.open(encoding="utf-8-sig", newline="") as f:
            rows = json.load(f) if args.questions.suffix.lower() == ".json" else list(csv.DictReader(f))
        if args.limit:
            rows = rows[:args.limit]
        if args.ids:
            rows = [row for row in rows if str(row["id"]) in args.ids]
            if len(rows) != len(set(args.ids)):
                ap.error("Не найдены некоторые ID")
        if bot.retriever.embedder is not None:
            # На машине с одной одновременно загруженной моделью заранее считаем
            # embeddings запросов, чтобы не переключать BGE/Qwen на каждом ходе.
            warmup = time.perf_counter()
            bot.retriever.prefetch([row["question"] for row in rows])
            report["query_warmup_s"] = round(time.perf_counter() - warmup, 2)
        bot.llm.check_connection()
        if args.unload_model:
            if config.LLM_PROVIDER != "ollama":
                raise LLMError("Выгрузка модели поддерживается только для локальной Ollama.", "configuration")
            response = bot.llm.client.post(config.OLLAMA_BASE_URL + "/api/generate",
                                           json={"model": config.OLLAMA_MODEL, "keep_alive": 0})
            response.raise_for_status()
        api = EnergytoolAI(bot)
        consecutive_errors = 0
        with tempfile.TemporaryDirectory(prefix="energytool-demo-") as sessions:
            with patch.object(config, "SESSION_DIR", sessions):
                for row in rows:
                    if any(str(r["id"]) == str(row["id"]) for r in report["results"]):
                        continue
                    turns = parse_dialog(row["question"])
                    state = ConversationState()
                    for role, text in turns[:-1]:
                        state.add(role, text)
                    started = time.perf_counter()
                    result = api.chat(turns[-1][1], session_id=state.session_id)
                    elapsed = round((time.perf_counter() - started) * 1000)
                    debug = result["debug"]
                    actual_mode = debug.get("retrieval", {}).get("mode")
                    report["results"].append({"id": row["id"], "question": row["question"],
                                               "answer": result["answer"], "status": result["status"],
                                               "sources": result["sources"], "elapsed_ms": elapsed,
                                               "actual_mode": actual_mode,
                                               "action": debug.get("action"), "reason_code": debug.get("reason_code"),
                                               "error_code": debug.get("error_code"),
                                               "decision_warnings": debug.get("decision_warnings", []),
                                               "context_ids": debug.get("context_ids", []),
                                               "timings_ms": debug.get("timings_ms", {}),
                                               "fallback": actual_mode is not None and actual_mode != args.mode})
                    report["results"][-1]["reranked"] = debug.get("retrieval", {}).get("reranked", False)
                    report["results"][-1]["rerank_error"] = debug.get("retrieval", {}).get("rerank_error")
                    for key in ("customer_facts", "selected_evidence", "claims", "question_slot", "answer_validation_errors",
                                "answer_task", "answer_contract", "requirement_checks", "quality", "llm_calls",
                                "decision", "decision_input_flags", "decision_input_types", "generation_attempts", "draft_claims", "composition_fallback"):
                        report["results"][-1][key] = debug.get(key)
                    save()
                    print(f"{row['id']}: {result['status']} {elapsed} ms", flush=True)
                    consecutive_errors = consecutive_errors + 1 if result["status"] == "error" else 0
                    if args.max_errors > 0 and consecutive_errors >= args.max_errors:
                        raise LLMError("Последовательные технические ошибки; прогон остановлен для диагностики.", "error_limit")
        timings = [r["elapsed_ms"] for r in report["results"]]
        report["summary"] = {"n": len(timings), "statuses": dict(Counter(r["status"] for r in report["results"])),
                             "extractive_fallback_count": sum(r["reason_code"] == "extractive_fallback" for r in report["results"]),
                             "fallback_count": sum(r["fallback"] for r in report["results"]),
                             "median_ms": statistics.median(timings) if timings else None,
                             "p95_ms": sorted(timings)[max(0, math.ceil(len(timings) * .95) - 1)] if timings else None}
        report["completed"] = True
        save()
        print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
        return 0
    except LLMError as exc:
        report["error"], report["error_code"] = str(exc), exc.code
        save()
        print(f"Прогон остановлен: {exc}", flush=True)
        return 2
    except Exception as exc:
        report["error"], report["error_code"] = type(exc).__name__, "execution_error"
        save()
        print(f"Ошибка прогона: {type(exc).__name__}", flush=True)
        return 2
    finally:
        if bot is not None:
            bot.close()


if __name__ == "__main__":
    raise SystemExit(main())
