"""Сохранение реальных диалогов проверки: JSON, UTF-8 TXT и JSON-сессии."""
import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import config
from consultant.consultant import EnergytoolConsultant
from integration.interface import EnergytoolAI
from llm.qwen import LLMError
from memory.conversation_state import ConversationState

GROUPS = ("basic", "paraphrases", "boundaries", "dialogues", "robustness")


def load_cases(path, group="all", case_ids=None):
    cases = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    seen = set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("Нужен непустой список сценариев")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or case["id"] in seen:
            raise ValueError("У сценария должен быть уникальный строковый id")
        seen.add(case["id"])
        if case.get("group") not in GROUPS or not isinstance(case.get("title"), str):
            raise ValueError("Неизвестная группа или отсутствует название")
        if not isinstance(case.get("turns"), list) or not case["turns"] or any(not isinstance(t, str) or not t.strip() for t in case["turns"]):
            raise ValueError("turns должен содержать непустые пользовательские реплики")
        for field in ("expected", "forbidden"):
            if not isinstance(case.get(field), list) or any(not isinstance(t, str) for t in case[field]):
                raise ValueError("expected и forbidden должны быть списками строк")
    if case_ids and not set(case_ids).issubset(seen):
        raise ValueError("Неизвестный ID сценария")
    chosen = [c for c in cases if (group == "all" or c["group"] == group) and (not case_ids or c["id"] in case_ids)]
    if not chosen:
        raise ValueError("Нет сценариев для выбранных условий")
    return chosen


def code_hash():
    paths = [config.ROOT / "config.py"] + [p for name in ("consultant", "rag", "llm", "memory", "integration", "prompts")
                                            for p in (config.ROOT / name).glob("*.py")]
    return hashlib.sha256(b"".join(str(p.relative_to(config.ROOT)).encode() + b"\0" + p.read_bytes() for p in sorted(paths))).hexdigest()


def save_report(output, report):
    """Прогресс сохраняется после каждого хода, даже при незавершённом прогоне."""
    output = Path(output)
    target = output / "results.json"
    pending = output / "results.json.tmp"
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(target)
    lines = ["Проверка EnergyTool", "Модель: " + str(report.get("model", "")),
             "Хеш кода: " + str(report.get("code_sha256", "")),
             "Завершено: " + str(report.get("completed", False)),
             "Завершение прогона не означает правильность всех ответов.", ""]
    for case in report.get("cases", []):
        lines += ["=" * 72, f"{case['id']} | {case['title']} | повтор {case['repeat']}",
                  "Ожидаемое поведение:"] + ["- " + text for text in case["expected"]]
        lines += ["Нельзя:"] + ["- " + text for text in case["forbidden"]]
        for turn in case["turns"]:
            lines += ["", "Вы: " + turn["message"], "EnergyTool: " + turn["answer"],
                      "Статус: " + turn["status"], "Время, мс: " + str(turn["elapsed_ms"]),
                      "Диагностика:", json.dumps({k: v for k, v in turn.items() if k not in {"message", "answer"}}, ensure_ascii=False, indent=2)]
    if report.get("error_code"):
        lines += ["", "Ошибка прогона: " + report["error_code"]]
    (output / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_cases(api, cases, repeats, report, output, max_errors=3):
    consecutive_errors = 0
    for repeat in range(1, repeats + 1):
        for case in cases:
            state = ConversationState()
            record = {k: case[k] for k in ("id", "group", "title", "expected", "forbidden")}
            record.update({"repeat": repeat, "completed": False, "turns": []})
            report["cases"].append(record)
            save_report(output, report)
            for number, message in enumerate(case["turns"], 1):
                started = time.perf_counter()
                if message == "/new":
                    state = state.reset()
                    result = {"answer": "Начат новый диалог.", "status": "session_reset", "sources": [],
                              "session_id": state.session_id, "debug": {"event": "session_reset"}}
                else:
                    try:
                        # Передаётся только реплика пользователя. Ожидания и
                        # запреты теста не попадают в LLM или историю клиента.
                        result = api.chat(message, session_id=state.session_id)
                    except Exception as exc:
                        result = {"answer": "Техническая ошибка при выполнении проверки.", "status": "error", "sources": [],
                                  "session_id": state.session_id,
                                  "debug": {"error_code": exc.code if isinstance(exc, LLMError) else type(exc).__name__}}
                    state = ConversationState(result["session_id"]).load()
                turn = {"number": number, "message": message, **result,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                        "state_snapshot": {k: state.data.get(k) for k in ("customer_facts", "asked_slots", "pending_handoff", "lead")}}
                record["turns"].append(turn)
                save_report(output, report)
                print(f"{case['id']} [{repeat}] {number}/{len(case['turns'])}: {result['status']}", flush=True)
                if message != "/new":
                    consecutive_errors = consecutive_errors + 1 if result["status"] == "error" else 0
                if max_errors and consecutive_errors >= max_errors:
                    raise LLMError("Прогон остановлен после последовательных технических ошибок.", "error_limit")
            record["completed"] = True
            save_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=config.ROOT.parent.parent / "bot_checks/scenarios.json")
    parser.add_argument("--group", choices=("all",) + GROUPS, default="basic")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--repeat", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output-root", type=Path, default=config.ROOT.parent.parent / "test-results")
    parser.add_argument("--list", action="store_true", help="Показать список без запуска модели")
    args = parser.parse_args()
    try:
        cases = load_cases(args.scenarios, args.group, args.ids)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.list:
        for case in cases:
            print(f"{case['id']}: {case['title']} ({len(case['turns'])} ходов)")
        return 0
    output = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.group + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "completed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "code_sha256": code_hash(), "scenario_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
              "provider": config.LLM_PROVIDER,
              "model": config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.LLM_MODEL,
              "python": platform.python_version(), "group": args.group, "repeats": args.repeat,
              "configuration": {"retriever_mode": config.RETRIEVER_MODE, "temperature": config.LLM_TEMPERATURE,
                                "max_tokens": config.MAX_TOKENS, "context_max_chars": config.CONTEXT_MAX_CHARS,
                                "ollama_think": config.OLLAMA_THINK,
                                "source_first_responses": config.SOURCE_FIRST_RESPONSES},
              "manual_quality_review_required": True, "cases": []}
    save_report(output, report)
    print("Результаты: " + str(output.resolve()), flush=True)
    previous_sessions = config.SESSION_DIR
    config.SESSION_DIR = str(output / "sessions")
    bot = None
    exit_code = 0
    try:
        bot = EnergytoolConsultant()
        bot.llm.check_connection()
        execute_cases(EnergytoolAI(bot), cases, args.repeat, report, output)
        statuses = Counter(turn["status"] for case in report["cases"] for turn in case["turns"])
        report["summary"] = {"statuses": dict(statuses), "cases": len(report["cases"]),
                             "technical_errors": statuses.get("error", 0)}
        report["completed"] = True
        exit_code = 2 if statuses.get("error") else 0
    except KeyboardInterrupt:
        report["error_code"] = "interrupted_by_user"
        exit_code = 130
    except Exception as exc:
        report["error_code"] = exc.code if isinstance(exc, LLMError) else type(exc).__name__
        exit_code = 2
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(output, report)
        config.SESSION_DIR = previous_sessions
        if bot is not None:
            bot.close()
    print("Сохранены results.json, results.txt и sessions в " + str(output.resolve()), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
