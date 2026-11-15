"""Терминальный чат. /debug включает безопасную диагностику."""
import json
import config
from consultant.consultant import EnergytoolConsultant
from llm.qwen import LLMError
from memory.conversation_state import ConversationState


def main():
    print("EnergyTool — консультант МТС")
    print("/new — новый диалог; /debug — диагностика; /exit — выход.\n")
    bot = None
    try:
        bot = EnergytoolConsultant()
        bot.llm.check_connection()
        model = config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.LLM_MODEL
        print(f"Провайдер: {config.LLM_PROVIDER}; модель: {model}; поиск: {bot.retriever.mode}")
        print(f"База знаний: {len(bot.retriever.chunks)} фрагментов.\n")
        session, debug_enabled = ConversationState(), config.DEBUG
        while True:
            try:
                message = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message == "/exit":
                break
            if message == "/new":
                session = session.reset()
                print("Начат новый диалог.")
                continue
            if message == "/debug":
                debug_enabled = not debug_enabled
                print("Диагностика " + ("включена." if debug_enabled else "выключена."))
                continue
            if not message:
                continue
            result = bot.reply(message, session)
            session = result["state"]
            print("\nEnergyTool:", result["answer"])
            d = result["debug"]
            if debug_enabled:
                r = d.get("retrieval", {})
                safe = {k: d.get(k) for k in ("action", "reason_code", "error_code", "decision_input_types",
                                             "decision_input_flags", "decision_warnings", "timings_ms",
                                             "elapsed_ms", "retrieval_called", "generation_called", "context_chars")}
                safe.update({"customer_facts": d.get("customer_facts"),
                             "answer_task": d.get("answer_task"), "answer_contract": d.get("answer_contract"),
                             "requirement_checks": d.get("requirement_checks"), "quality": d.get("quality"),
                             "selected_evidence": d.get("selected_evidence"),
                             "question_slot": d.get("question_slot"),
                             "composition_fallback": d.get("composition_fallback"),
                             "answer_validation_errors": d.get("answer_validation_errors")})
                safe.update({"mode": r.get("mode"), "confidence": r.get("confidence"),
                             "no_answer": r.get("no_answer"), "context_ids": d.get("context_ids", [])})
                print(json.dumps(safe, ensure_ascii=False, indent=2))
    except (LLMError, RuntimeError, ValueError) as exc:
        print(f"Ошибка запуска: {exc}")
    finally:
        if bot is not None:
            bot.close()


if __name__ == "__main__":
    main()
