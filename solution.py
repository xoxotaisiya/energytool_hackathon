"""Простой Python API для ядра Energytool."""

from consultant.consultant import EnergytoolConsultant
from memory.conversation_state import ConversationState


def build_consultant():
    """Создает консультанта с настроенными RAG и LLM."""
    return EnergytoolConsultant()


def answer(message, history=None):
    """Отвечает на один вопрос с учетом переданной истории."""
    bot = build_consultant()
    session = ConversationState()

    for item in history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            session.add(role, content)

    return bot.reply(message, session)["answer"]
