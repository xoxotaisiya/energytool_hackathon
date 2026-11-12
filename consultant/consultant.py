"""Оркестратор: анализ потребности -> уточнение -> RAG -> grounded answer."""
import time

import config
from llm.qwen import LLMClient, LLMError
from memory.conversation_state import ConversationState
from rag.retrieval import Retriever
from prompts.system_prompts import CONSULTANT_ROLE, DECISION_PROMPT, ANSWER_PROMPT


class EnergytoolConsultant:
    def __init__(self, retriever=None, llm=None):
        """Настраивает языковую модель и поиск по базе знаний."""
        self.llm = llm or LLMClient()
        if retriever is None:
            kb = config.KB_PATH or None
            mode = None if config.RETRIEVER_MODE == "auto" else config.RETRIEVER_MODE
            retriever = Retriever(kb_path=kb, mode=mode)
        self.retriever = retriever

    def _decision(self, history):
        """Определяет задачу клиента и проверяет, нужны ли уточнения."""
        messages = [
            {"role": "system", "content": CONSULTANT_ROLE + "\n\n" + DECISION_PROMPT},
            {"role": "user", "content": "История диалога:\n" + self._format_history(history)}
        ]
        data = self.llm.chat_json(messages)
        # Защита от некорректной формы JSON.
        return {
            "intent": str(data.get("intent", "other")),
            "customer_goal": str(data.get("customer_goal", "")),
            "known_facts": data.get("known_facts", []) if isinstance(data.get("known_facts", []), list) else [],
            "missing_info": data.get("missing_info", []) if isinstance(data.get("missing_info", []), list) else [],
            "clarification_question": str(data.get("clarification_question", "")).strip(),
            "out_of_scope": bool(data.get("out_of_scope", False)),
            "needs_retrieval": bool(data.get("needs_retrieval", True)),
            "wants_human": bool(data.get("wants_human", False)),
        }

    @staticmethod
    def _format_history(history):
        """Преобразует историю сообщений в текст для модели."""
        return "\n".join(("Клиент: " if x["role"] == "user" else "Консультант: ") + x["content"]
                         for x in history[-12:])

    def reply(self, user_text, state=None):
        """Обрабатывает сообщение клиента и возвращает ответ с отладочными данными."""
        started = time.perf_counter()
        state = state or ConversationState()
        if not user_text.strip():
            return {"answer": "Напишите, пожалуйста, какую задачу вы хотите решить.",
                    "state": state, "debug": {}}

        state.add("user", user_text)
        history = state.history()
        try:
            decision = self._decision(history)
        except LLMError as exc:
            answer = ("Сейчас не удалось связаться с языковой моделью. "
                      "Проверьте её запуск и настройки, затем повторите вопрос.")
            state.add("assistant", answer, {"error": str(exc)})
            return {"answer": answer, "state": state, "debug": {"error": str(exc)}}

        if decision["out_of_scope"] or decision["intent"] == "out_of_scope":
            answer = ("Я консультирую по продуктам МТС Energytool и связанным задачам "
                      "учёта, энергозатрат и выявления потерь. По этому вопросу лучше "
                      "обратиться к профильному специалисту.")
            state.add("assistant", answer, {"intent": decision["intent"]})
            return {"answer": answer, "state": state,
                    "debug": {"decision": decision, "elapsed_ms": round((time.perf_counter()-started)*1000)}}

        if decision["wants_human"]:
            answer = "Конечно. Могу подготовить краткое резюме для специалиста. Оставьте, пожалуйста, ФИО, телефон и название компании."
            state.add("assistant", answer, {"stage": "handoff_offer"})
            return {"answer": answer, "state": state, "debug": {"decision": decision}}

        # Retrieval использует полный контекст диалога; существующий retrieval.py умеет парсить историю.
        retrieval = self.retriever.retrieve(user_text, history=history[:-1])
        chunks = retrieval.get("chunks", [])
        guards = retrieval.get("guards", [])
        if retrieval.get("no_answer") or not chunks:
            # Если поиск не помог, один конкретный вопрос может уточнить задачу.
            has_question = bool(decision["missing_info"] and decision["clarification_question"])
            answer = decision["clarification_question"] if has_question else (
                "В доступной базе знаний я не нашёл подтверждения, достаточного для точного ответа. "
                "Не хочу предполагать характеристики или обещать неподтверждённый результат. "
                "Предлагаю обсудить задачу со специалистом Energytool."
            )
            stage = "clarification" if has_question else "insufficient_evidence"
            state.add("assistant", answer, {"stage": stage, "retrieval_mode": retrieval.get("mode")})
            return {"answer": answer, "state": state,
                    "debug": {"decision": decision, "retrieval": retrieval, "stage": stage,
                              "elapsed_ms": round((time.perf_counter()-started)*1000)}}

        context = []
        for c in chunks[:5]:
            context.append(f"[{c.get('id')} | {c.get('path')}]\n{c.get('text','')}")
        for g in guards[:4]:
            context.append(f"[ОГРАНИЧЕНИЕ {g.get('id')} | {g.get('path')}]\n{g.get('text','')}")
        messages = [
            {"role": "system", "content": CONSULTANT_ROLE + "\n\n" + ANSWER_PROMPT},
            {"role": "user", "content": (
                "История диалога:\n" + self._format_history(history) +
                "\n\nЗадача клиента, определённая предварительно:\n" + decision["customer_goal"] +
                "\n\nФрагменты базы знаний Energytool:\n" + "\n\n".join(context) +
                "\n\nОтветь на последнюю реплику клиента. Не добавляй фактов вне контекста."
            )}
        ]
        try:
            answer = self.llm.chat(messages)
        except LLMError as exc:
            answer = (
                "Не удалось получить ответ от языковой модели. "
                "Проверьте подключение к Ollama и повторите запрос. "
                "Найденные материалы не были выданы как готовая консультация."
            )
            state.add("assistant", answer, {"stage": "llm_error", "error": str(exc)})
            return {
                "answer": answer,
                "state": state,
                "debug": {
                    "decision": decision,
                    "retrieval": retrieval,
                    "error": str(exc),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                },
            }
        source_ids = [c.get("id") for c in chunks]
        state.add("assistant", answer, {"stage": "rag_answer", "sources": source_ids})
        debug = {"decision": decision, "retrieval": retrieval,
                 "elapsed_ms": round((time.perf_counter()-started)*1000)}
        return {"answer": answer, "state": state, "debug": debug}
