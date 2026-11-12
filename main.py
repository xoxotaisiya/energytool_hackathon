"""Запуск терминального чата Energytool."""

from consultant.consultant import EnergytoolConsultant
from llm.qwen import LLMError
from memory.conversation_state import ConversationState


def main():
    """Проверяет подключение к Qwen и запускает терминальный диалог."""
    print("Energytool — AI-консультант МТС Energytool")
    print("Команды: /new — новый диалог; /exit — выход.\n")

    bot = EnergytoolConsultant()
    try:
        bot.llm.check_connection()
    except LLMError as exc:
        print(f"Ошибка запуска: {exc}")
        return

    print(f"Модель: {bot.llm.__class__.__name__}; поиск: {bot.retriever.mode}")
    print(f"База знаний: {len(bot.retriever.chunks)} фрагментов.\n")
    session = ConversationState()

    while True:
        try:
            message = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nРабота завершена.")
            break

        if message == "/exit":
            break
        if message == "/new":
            session = session.reset()
            print("Начат новый диалог.")
            continue
        if not message:
            continue

        result = bot.reply(message, session)
        session = result["state"]
        print("\nEnergytool:", result["answer"])

        debug = result.get("debug", {})
        retrieval = debug.get("retrieval", {})
        chunks = retrieval.get("chunks", [])
        if chunks:
            print("Источники:")
            for chunk in chunks:
                print(f"  - {chunk.get('id')} — {chunk.get('path')} [{chunk.get('file', '')}]")
        if debug.get("error"):
            print("Техническая ошибка:", debug["error"])


if __name__ == "__main__":
    main()
