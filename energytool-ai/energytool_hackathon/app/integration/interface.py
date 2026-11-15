"""Публичный интерфейс: статус определяется принятым действием консультанта."""
from uuid import UUID
from consultant.consultant import EnergytoolConsultant
from memory.conversation_state import ConversationState


class EnergytoolAI:
    def __init__(self, consultant=None):
        self.consultant = consultant if consultant is not None else EnergytoolConsultant()

    def chat(self, message, session_id=None):
        state = self._get_session(session_id)
        result = self.consultant.reply(message, state)
        debug = result.get("debug", {})
        retrieval = debug.get("retrieval", {})
        action = debug.get("action")
        status = self._get_status(debug, debug.get("decision", {}), debug.get("stage"))
        used = set(debug.get("source_ids", []))
        sources = [{"id": c.get("id"), "path": c.get("path"), "file": c.get("file")}
                   for c in retrieval.get("chunks", []) + retrieval.get("supporting_chunks", []) + retrieval.get("guards", [])
                   if c.get("id") in used]
        return {"answer": result["answer"], "status": status,
                "question": debug.get("question"),
                "sources": sources, "session_id": result["state"].session_id, "debug": debug}

    @staticmethod
    def _get_session(session_id):
        if not session_id:
            return ConversationState()
        try:
            session_id = str(UUID(str(session_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("session_id должен быть UUID.") from exc
        return ConversationState(session_id).load()

    @staticmethod
    def _get_status(debug, decision, stage):
        mapping = {"answered": "answered", "smalltalk": "answered",
                   "clarification": "clarification_needed", "handoff": "handoff",
                   "no_evidence": "no_evidence", "out_of_scope": "out_of_scope", "error": "error"}
        action = debug.get("action") or stage
        if action in mapping:
            return mapping[action]
        # Совместимость с консультантами предыдущей версии.
        if debug.get("error"):
            return "error"
        if stage == "insufficient_evidence":
            return "no_evidence"
        if decision.get("out_of_scope") is True or decision.get("intent") == "out_of_scope":
            return "out_of_scope"
        if stage == "handoff_offer" or decision.get("wants_human") is True:
            return "handoff"
        return "answered"

    def close(self):
        self.consultant.close()
