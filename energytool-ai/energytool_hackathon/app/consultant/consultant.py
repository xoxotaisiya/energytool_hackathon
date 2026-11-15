"""Оркестрация консультации с явными действиями и диагностикой."""
import json
import re
import time

import config
from llm.qwen import LLMClient, LLMError
from memory.conversation_state import ConversationState
from rag.retrieval import Retriever
from prompts.system_prompts import CONSULTANT_ROLE, DECISION_PROMPT, ANSWER_PROMPT, PRODUCT_SCOPE
from consultant.contracts import validate_decision, validate_answer, fallback_decision
from consultant.grounded import reply_grounded
from consultant.dialogue import human_request, human_declined, cancels_pending_handoff, customer_facts

SMALLTALK = {
    "greeting": (r"(здравствуйте|здравствуй|привет|добрый день|добрый вечер|доброе утро)[!. ]*",
                 "Здравствуйте! Какую задачу вы хотите решить: снизить энергозатраты, наладить учёт или найти потери?"),
    "thanks": (r"(спасибо|благодарю|спасибо большое|большое спасибо)[!. ]*",
               "Пожалуйста! Если появятся вопросы по EnergyTool, обращайтесь."),
    "goodbye": (r"(до свидания|пока|всего доброго)[!. ]*", "До свидания!"),
}


class EnergytoolConsultant:
    def __init__(self, retriever=None, llm=None):
        self.llm = llm if llm is not None else LLMClient()
        self.retriever = retriever if retriever is not None else Retriever(
            kb_path=config.KB_PATH or None, mode=config.RETRIEVER_MODE)

    @staticmethod
    def _format_history(history):
        return "\n".join(("Клиент: " if x["role"] == "user" else "Консультант: ") + x["content"]
                         for x in history[-12:])

    @staticmethod
    def _context(retrieval):
        # Сохраняем целые фрагменты, ограничиваем общий объём, удаляем повторные ID.
        chunks, guards, seen = [], [], set()
        guard_budget = min(4000, config.CONTEXT_MAX_CHARS // 3)
        remaining = config.CONTEXT_MAX_CHARS
        for items, output, budget in ((retrieval.get("guards", []), guards, guard_budget),
                                      (retrieval.get("supporting_chunks", []) + retrieval.get("chunks", []), chunks, None)):
            used = 0
            for c in items:
                text = f"[{c['id']} | {c.get('path', '')}]\n{c.get('text', '')}"
                cost = len(text) + 2
                if c["id"] in seen or cost > remaining:
                    continue
                if budget is not None and used + cost > budget:
                    continue
                output.append((c, text))
                seen.add(c["id"])
                remaining -= cost
                used += cost
        context = "\n\n".join(text for _, text in chunks + guards)
        return context, [c["id"] for c, _ in chunks + guards]

    def _decision(self, history, context, debug):
        context = context[:3000]
        debug["decision_context_chars"] = len(context)
        raw = self.llm.chat_json([
            {"role": "system", "content": PRODUCT_SCOPE + "\n\n" + DECISION_PROMPT},
            {"role": "user", "content": "История:\n" + self._format_history(history) +
             "\n\nНайденные материалы для предварительного анализа:\n" + context},
        ])
        if isinstance(raw, dict):
            # Только известные поля; не копируем произвольный ответ модели в журналы.
            keys = ("intent", "out_of_scope", "needs_retrieval", "wants_human")
            debug["decision_input_types"] = {k: type(raw.get(k)).__name__ for k in keys}
            debug["decision_input_flags"] = {k: raw.get(k)[:200] if isinstance(raw.get(k), str) else raw.get(k) for k in keys
                                              if isinstance(raw.get(k), (str, bool, type(None)))}
        decision, warnings = validate_decision(raw)
        debug["decision_warnings"] = warnings
        return decision

    def reply(self, user_text, state=None):
        started = time.perf_counter()
        state = state or ConversationState()
        debug = {"timings_ms": {}, "retrieval_called": False, "generation_called": False}
        if hasattr(self.llm, "reset_metrics"):
            self.llm.reset_metrics()

        def finish(answer, action, reason, question=None, source_ids=None):
            if hasattr(self.llm, "metrics"):
                debug["llm_calls"] = self.llm.metrics()
            debug.update({"action": action, "stage": action, "reason_code": reason,
                          "question": question, "source_ids": source_ids or [],
                          "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            state.add("assistant", answer, debug)
            return {"answer": answer, "state": state, "debug": debug}

        if not isinstance(user_text, str):
            raise ValueError("message должен быть строкой")
        if not user_text.strip():
            return finish("Напишите, пожалуйста, какую задачу вы хотите решить.", "clarification", "empty_input",
                          "Какую задачу вы хотите решить?")
        state.add("user", user_text.strip())
        history = state.history()
        if state.data.get("pending_handoff") and cancels_pending_handoff(user_text):
            state.data["pending_handoff"] = False
            if not re.search(r"\?|расскажите|объясните|сравните|подскажите", user_text, re.I):
                return finish("Хорошо, продолжим здесь. Какой вопрос хотите обсудить?", "smalltalk", "handoff_cancelled")
        if human_declined(user_text):
            state.data["pending_handoff"] = False
        if human_request(user_text):
            state.data["pending_handoff"] = True
            state.data["customer_facts"] = customer_facts(history)
            debug["customer_facts"] = state.data["customer_facts"]
            debug["decision"] = {"wants_human": True, "intent": "human_request"}
            return finish("Могу подготовить обращение специалисту. Оставьте удобный контакт; имя и компанию можно добавить по желанию.",
                          "handoff", "user_requested_human")
        if state.data.get("pending_handoff"):
            contacts = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\w)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)", user_text)
            if contacts:
                state.data["lead"] = {"contacts": list(dict.fromkeys(contacts)), "status": "saved_locally",
                                      "customer_facts": customer_facts(history), "source": "user"}
                state.data["pending_handoff"] = False
                return finish("Контакт сохранён в этой сессии для обращения к специалисту.", "handoff", "contact_saved")
        for kind, (pattern, answer) in SMALLTALK.items():
            if re.fullmatch(pattern, user_text.strip(), flags=re.IGNORECASE):
                return finish(answer, "smalltalk", kind)

        t0 = time.perf_counter()
        try:
            debug["retrieval_called"] = True
            retrieval = self.retriever.retrieve(user_text, history=history[:-1])
        except Exception as exc:
            debug["error"] = "Не удалось выполнить поиск."
            debug["error_code"] = "retrieval_error"
            debug["error_type"] = type(exc).__name__
            return finish("Сейчас не удалось выполнить поиск по базе знаний. Повторите запрос.",
                          "error", "retrieval_error")
        debug["timings_ms"]["retrieval"] = round((time.perf_counter() - t0) * 1000)
        debug["retrieval"] = retrieval
        context, available_ids = self._context(retrieval)
        debug["context_ids"], debug["context_chars"] = available_ids, len(context)

        if config.GROUNDED_ANSWERS:
            return reply_grounded(self, history, retrieval, available_ids, state, debug, finish)

        t0 = time.perf_counter()
        try:
            decision = self._decision(history, context, debug)
        except LLMError as exc:
            if exc.code in {"json_format", "decision_schema"}:
                # Ошибка схемы не становится отказом. Финальная модель увидит источники.
                decision = fallback_decision()
                debug["decision_warnings"] = [exc.code]
            else:
                debug["error"], debug["error_code"] = str(exc), exc.code
                return finish("Сейчас не удалось получить ответ модели. Проверьте подключение и повторите вопрос.",
                              "error", exc.code)
        debug["timings_ms"]["decision"] = round((time.perf_counter() - t0) * 1000)
        debug["decision"] = decision

        if decision["wants_human"]:
            if not human_request(user_text):
                decision["wants_human"] = False
        if decision["wants_human"]:
            return finish("Могу подготовить обращение специалисту. Оставьте удобный контакт; имя и компанию можно добавить по желанию.",
                          "handoff", "user_requested_human")
        if not available_ids:
            if decision["missing_info"] and decision["clarification_question"]:
                question = decision["clarification_question"]
                return finish(question, "clarification", "empty_retrieval_clarify", question)
            if decision["out_of_scope"] or decision["intent"] == "out_of_scope":
                return finish("Я консультирую по продуктам EnergyTool и связанным задачам энергозатрат и учёта. "
                              "Этот вопрос выходит за рамки доступной базы.", "out_of_scope", "empty_retrieval_out_of_scope")
            return finish("В доступной базе не нашлось подтверждения для ответа. Могу предложить обсудить вопрос со специалистом EnergyTool.",
                          "no_evidence", "empty_retrieval")

        messages = [
            {"role": "system", "content": CONSULTANT_ROLE + "\n\n" + ANSWER_PROMPT},
            {"role": "user", "content": "История:\n" + self._format_history(history) +
             "\n\nПредварительный анализ (может быть ошибочным):\n" +
             json.dumps(decision, ensure_ascii=False) +
             "\n\nФрагменты базы:\n" + context +
             "\n\nОтветь на последнюю реплику клиента."},
        ]
        t0 = time.perf_counter()
        try:
            debug["generation_called"] = True
            debug["generation_attempts"] = 0
            for attempt in range(2):
                debug["generation_attempts"] += 1
                try:
                    output = validate_answer(self.llm.chat_json(messages), set(available_ids))
                    break
                except LLMError as exc:
                    if attempt or exc.code not in {"json_format", "answer_schema", "decision_schema"}:
                        raise
                    messages.append({"role": "user", "content": "Исправь формат: верни только JSON по заданной схеме. "
                                     "Для answered укажи source_ids из переданного контекста; если прямого ответа нет, выбери no_evidence."})
            debug["timings_ms"]["generation"] = round((time.perf_counter() - t0) * 1000)
        except LLMError as exc:
            debug["timings_ms"]["generation"] = round((time.perf_counter() - t0) * 1000)
            debug["error"], debug["error_code"] = str(exc), exc.code
            return finish("Не удалось обработать ответ модели. Повторите запрос; если ошибка сохраняется, проверьте настройки модели.",
                          "error", exc.code)
        reasons = {"answered": "model_answered_with_sources", "clarification": "missing_customer_details",
                   "no_evidence": "model_no_direct_evidence", "out_of_scope": "model_out_of_scope",
                   "handoff": "model_handoff"}
        return finish(output["answer"], output["action"], reasons[output["action"]],
                      output["question"] or None, output["source_ids"])

    def close(self):
        close = getattr(self.llm, "close", None)
        if close:
            close()
        close = getattr(self.retriever, "close", None)
        if close:
            close()
