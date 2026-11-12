"""Consultant orchestration: intent, RAG, grounded answer and lead collection."""

from __future__ import annotations

import re
import time

import config
from llm.qwen import LLMClient, LLMError
from memory.conversation_state import ConversationState
from prompts.system_prompts import ANSWER_PROMPT, CONSULTANT_ROLE
from rag.retrieval import Retriever


HANDOFF_PATTERN = re.compile(
    r"\b(менеджер|специалист|оператор|заявк|перезвон|связаться|оставить контакты)\w*\b",
    re.IGNORECASE,
)
GREETING_PATTERN = re.compile(
    r"^(привет|здравствуй(?:те)?|добрый (?:день|вечер|утро)|hi|hello)[!,.\s]*$",
    re.IGNORECASE,
)


class EnergytoolConsultant:
    def __init__(self, retriever=None, llm=None):
        self.llm = llm or LLMClient()
        if retriever is None:
            kb = config.KB_PATH or None
            mode = None if config.RETRIEVER_MODE == "auto" else config.RETRIEVER_MODE
            retriever = Retriever(kb_path=kb, mode=mode)
        self.retriever = retriever

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        return "\n".join(
            ("Клиент: " if item["role"] == "user" else "Консультант: ") + item["content"]
            for item in history[-12:]
        )

    @staticmethod
    def _start_lead(state: ConversationState) -> dict:
        lead = {
            "status": "collecting",
            "pending_field": "full_name",
            "full_name": None,
            "phone": None,
            "company": None,
            "summary": None,
            "created_at": state._now(),
        }
        state.set_lead(lead)
        return lead

    @staticmethod
    def _normalise_phone(value: str) -> str | None:
        digits = re.sub(r"\D", "", value)
        if len(digits) == 11 and digits.startswith("8"):
            digits = "7" + digits[1:]
        elif len(digits) == 10:
            digits = "7" + digits
        if len(digits) != 11 or not digits.startswith("7"):
            return None
        return "+" + digits

    @staticmethod
    def _clean_name(value: str) -> str:
        value = re.sub(
            r"^(?:меня зовут|мо[её] фио|фио)\s*[:—-]?\s*", "", value.strip(), flags=re.IGNORECASE
        )
        return " ".join(value.split())

    @staticmethod
    def _build_summary(state: ConversationState, lead: dict) -> str:
        user_messages = []
        for message in state.data.get("messages", []):
            if message.get("role") != "user":
                continue
            text = message.get("content", "").strip()
            if text and not HANDOFF_PATTERN.search(text):
                user_messages.append(text)
        problem_messages = user_messages[:-3] if len(user_messages) > 3 else []
        problem = " ".join(problem_messages[-5:]).strip()
        if not problem:
            problem = "Клиент запросил консультацию специалиста; задача требует уточнения."
        return f"Клиент: {lead['full_name']}. Компания: {lead['company']}. Запрос: {problem[:900]}"

    def _lead_reply(self, user_text: str, state: ConversationState, started: float) -> dict:
        lead = dict(state.data.get("lead") or {})
        pending = lead.get("pending_field")
        if pending == "full_name":
            name = self._clean_name(user_text)
            if len(name.split()) < 2 or any(char.isdigit() for char in name):
                return self._save_simple(
                    state, "Укажите, пожалуйста, фамилию, имя и отчество полностью.", "lead_collecting", started
                )
            lead["full_name"] = name
            lead["pending_field"] = "phone"
            state.set_lead(lead)
            return self._save_simple(
                state, "Спасибо. Укажите контактный телефон в формате +7…", "lead_collecting", started
            )
        if pending == "phone":
            phone = self._normalise_phone(user_text)
            if not phone:
                return self._save_simple(
                    state,
                    "Не удалось распознать номер. Укажите телефон в формате +7 999 123-45-67.",
                    "lead_collecting",
                    started,
                )
            lead["phone"] = phone
            lead["pending_field"] = "company"
            state.set_lead(lead)
            return self._save_simple(state, "Как называется ваша компания?", "lead_collecting", started)
        if pending == "company":
            company = " ".join(user_text.strip().split())
            if len(company) < 2:
                return self._save_simple(
                    state, "Укажите, пожалуйста, название компании.", "lead_collecting", started
                )
            lead["company"] = company
            lead["pending_field"] = None
            lead["status"] = "complete"
            lead["completed_at"] = state._now()
            lead["summary"] = self._build_summary(state, lead)
            state.set_lead(lead)
            return self._save_simple(
                state,
                "Заявка сформирована. Специалист получит контакты и резюме диалога. "
                "В прототипе заявка сохранена локально и не отправляется во внешнюю CRM.",
                "lead_complete",
                started,
            )
        return self._save_simple(
            state,
            "Заявка уже сформирована. Для новой задачи начните новый диалог.",
            "lead_complete",
            started,
        )

    @staticmethod
    def _save_simple(state: ConversationState, answer: str, stage: str, started: float) -> dict:
        state.add("assistant", answer, {"stage": stage})
        return {
            "answer": answer,
            "state": state,
            "debug": {"stage": stage, "elapsed_ms": round((time.perf_counter() - started) * 1000)},
        }

    def reply(self, user_text: str, state: ConversationState | None = None) -> dict:
        started = time.perf_counter()
        state = state or ConversationState()
        user_text = user_text.strip()
        if not user_text:
            return {"answer": "Напишите, пожалуйста, какую задачу вы хотите решить.", "state": state, "debug": {"stage": "empty"}}

        state.add("user", user_text)
        lead = state.data.get("lead") or {}
        if lead.get("status") == "collecting":
            return self._lead_reply(user_text, state, started)
        if GREETING_PATTERN.match(user_text):
            return self._save_simple(
                state,
                "Здравствуйте! Расскажите, какую задачу по учёту энергоресурсов, "
                "контролю потребления или поиску потерь вы хотите решить?",
                "greeting",
                started,
            )
        if HANDOFF_PATTERN.search(user_text):
            self._start_lead(state)
            return self._save_simple(
                state,
                "Подготовлю заявку для специалиста. Укажите, пожалуйста, ваши ФИО. "
                "Используйте только тестовые данные.",
                "lead_collecting",
                started,
            )

        history = state.history()
        # Retrieve first, then let one model call both classify the request and answer it.
        # This halves model round-trips compared with the original decision -> answer chain.
        retrieval = self.retriever.retrieve(user_text, history=history[:-1])
        chunks = retrieval.get("chunks", [])
        guards = retrieval.get("guards", [])
        if retrieval.get("no_answer") or not chunks:
            answer = (
                "В базе знаний нет достаточной информации для точного ответа. "
                "Уточните, пожалуйста, задачу, тип объекта и что именно хотите контролировать. "
                "Если нужных сведений всё равно не окажется, предложу связаться со специалистом."
            )
            stage = "insufficient_evidence"
            state.add("assistant", answer, {"stage": stage})
            return {
                "answer": answer,
                "state": state,
                "debug": {"retrieval": retrieval, "stage": stage, "elapsed_ms": round((time.perf_counter() - started) * 1000)},
            }

        context = [f"[{c.get('id')} | {c.get('path')}]\n{c.get('text', '')}" for c in chunks[:5]]
        context.extend(
            f"[ОГРАНИЧЕНИЕ {g.get('id')} | {g.get('path')}]\n{g.get('text', '')}" for g in guards[:4]
        )
        messages = [
            {
                "role": "system",
                "content": CONSULTANT_ROLE + "\n\n" + ANSWER_PROMPT + (
                    "\n\nВерни только JSON-объект с полями answer и status. "
                    "status — одно из: answered, clarification_needed, out_of_scope, handoff. "
                    "Если запрос неясен — задай один конкретный вопрос. Если он вне компетенции "
                    "или контекст не подтверждает ответ — прямо скажи об этом. Ответ обычно до 150 слов."
                ),
            },
            {
                "role": "user",
                "content": (
                    "История диалога:\n" + self._format_history(history)
                    + "\n\nФрагменты базы знаний Energytool:\n" + "\n\n".join(context)
                    + "\n\nОтветь на последнюю реплику. Не добавляй фактов вне контекста."
                ),
            },
        ]
        try:
            response = self.llm.chat_json(messages)
            answer = str(response.get("answer", "")).strip()
            status = str(response.get("status", "answered")).strip()
            if not answer:
                raise LLMError("Модель вернула пустой ответ.")
            if status not in {"answered", "clarification_needed", "out_of_scope", "handoff"}:
                status = "answered"
        except LLMError as exc:
            answer = "Не удалось получить ответ от языковой модели. Проверьте подключение к Ollama и повторите запрос."
            state.add("assistant", answer, {"stage": "llm_error", "error": str(exc)})
            return {
                "answer": answer,
                "state": state,
                "debug": {"retrieval": retrieval, "error": str(exc), "stage": "llm_error", "elapsed_ms": round((time.perf_counter() - started) * 1000)},
            }

        if status == "handoff":
            self._start_lead(state)
            answer = "Подготовлю заявку для специалиста. Укажите, пожалуйста, ваши ФИО."
            stage = "lead_collecting"
        elif status == "clarification_needed":
            stage = "clarification"
        elif status == "out_of_scope":
            stage = "out_of_scope"
        else:
            stage = "rag_answer"
        decision = {"intent": status, "out_of_scope": status == "out_of_scope", "wants_human": status == "handoff"}
        source_ids = [chunk.get("id") for chunk in chunks]
        state.add("assistant", answer, {"stage": stage, "sources": source_ids})
        return {
            "answer": answer,
            "state": state,
            "debug": {"decision": decision, "retrieval": retrieval, "stage": stage, "elapsed_ms": round((time.perf_counter() - started) * 1000)},
        }
