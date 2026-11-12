"""Простой интерфейс AI-модуля для подключения к backend."""

from uuid import UUID

from consultant.consultant import EnergytoolConsultant
from memory.conversation_state import ConversationState


class EnergytoolAI:
    """Принимает сообщение и возвращает готовый ответ с метаданными."""

    def __init__(self, consultant=None):
        """Создает интерфейс и подключает консультанта."""
        self.consultant = consultant or EnergytoolConsultant()

    def chat(self, message, session_id=None):
        """Обрабатывает сообщение в указанной сессии и возвращает словарь."""
        state = self._get_session(session_id)
        result = self.consultant.reply(message, state)

        debug = result.get("debug", {})
        retrieval = debug.get("retrieval", {})
        decision = debug.get("decision", {})
        stage = debug.get("stage")

        status = self._get_status(debug, decision, stage)
        sources = [
            {"id": item.get("id"), "path": item.get("path"), "file": item.get("file")}
            for item in retrieval.get("chunks", [])
        ]

        return {
            "answer": result["answer"],
            "status": status,
            "question": result["answer"] if status == "clarification_needed" else None,
            "sources": sources,
            "session_id": result["state"].session_id,
            "lead": result["state"].data.get("lead"),
            "debug": debug,
        }

    @staticmethod
    def _get_session(session_id):
        """Загружает старую сессию или создает новую."""
        if not session_id:
            return ConversationState()

        # Не позволяем передать путь к файлу вместо ID сессии.
        try:
            session_id = str(UUID(str(session_id)))
        except ValueError as exc:
            raise ValueError("session_id должен быть UUID.") from exc

        return ConversationState(session_id).load()

    @staticmethod
    def _get_status(debug, decision, stage):
        """Определяет статус ответа для backend."""
        if debug.get("error"):
            return "error"
        if stage == "clarification":
            return "clarification_needed"
        if stage in {"handoff_offer", "lead_collecting", "lead_complete"} or decision.get("wants_human"):
            return "handoff"
        if stage == "insufficient_evidence":
            return "no_evidence"
        if decision.get("out_of_scope") or decision.get("intent") == "out_of_scope":
            return "out_of_scope"
        return "answered"
