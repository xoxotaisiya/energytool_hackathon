"""Простой Python API для ядра Energytool."""

from consultant.consultant import EnergytoolConsultant
from memory.conversation_state import ConversationState
from functools import lru_cache
import atexit


@lru_cache(maxsize=1)
def build_consultant():
    """Создает консультанта с настроенными RAG и LLM."""
    return EnergytoolConsultant()


def close_consultant():
    if build_consultant.cache_info().currsize:
        build_consultant().close()
        build_consultant.cache_clear()


atexit.register(close_consultant)


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
