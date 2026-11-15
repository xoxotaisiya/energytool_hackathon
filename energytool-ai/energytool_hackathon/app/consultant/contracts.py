"""Проверка решений модели без неявного bool('false')."""
from llm.qwen import LLMError

INTENTS = {"product_advice", "product_question", "energy_problem", "out_of_scope", "greeting", "other"}
ACTIONS = {"answered", "clarification", "handoff", "no_evidence", "out_of_scope"}


def _text(value, field, code="decision_schema"):
    if not isinstance(value, str):
        raise LLMError(f"Поле {field} должно быть строкой.", code)
    return value.strip()


def _strings(value, field, code="decision_schema"):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise LLMError(f"Поле {field} должно быть списком строк.", code)
    return [x.strip() for x in value if x.strip()]


def validate_decision(data):
    if not isinstance(data, dict):
        raise LLMError("Решение должно быть объектом.", "decision_schema")
    intent = data.get("intent")
    if not isinstance(intent, str) or intent not in INTENTS:
        raise LLMError("Неизвестный intent.", "decision_schema")
    result = {"intent": intent}
    warnings = []
    for field, default in (("out_of_scope", False), ("needs_retrieval", True), ("wants_human", False)):
        value = data.get(field, default)
        if type(value) is not bool:
            # Поддерживаем только явные true/false, а не произвольные truthy-значения.
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                value = value.strip().lower() == "true"
                warnings.append(f"normalized_boolean:{field}")
            else:
                raise LLMError(f"Поле {field} должно быть boolean.", "decision_schema")
        result[field] = value
    for field in ("customer_goal", "clarification_question"):
        result[field] = _text(data.get(field, ""), field)
    for field in ("known_facts", "missing_info"):
        result[field] = _strings(data.get(field, []), field)
    if intent == "out_of_scope" and not result["out_of_scope"]:
        warnings.append("conflicting_scope_flags")
    return result, warnings


def fallback_decision():
    return {"intent": "other", "customer_goal": "", "known_facts": [],
            "missing_info": [], "clarification_question": "", "out_of_scope": False,
            "needs_retrieval": True, "wants_human": False}


def validate_answer(data, available_ids):
    if not isinstance(data, dict) or not isinstance(data.get("action"), str) or data["action"] not in ACTIONS:
        raise LLMError("Ответ должен содержать допустимый action.", "answer_schema")
    action = data["action"]
    answer = _text(data.get("answer"), "answer", "answer_schema")
    question = _text(data.get("question", ""), "question", "answer_schema")
    source_ids = _strings(data.get("source_ids", []), "source_ids", "answer_schema")
    if not answer or len(answer) > 10000:
        raise LLMError("Пустой или слишком длинный ответ.", "answer_schema")
    if not set(source_ids).issubset(available_ids):
        raise LLMError("В ответе указаны источники вне контекста.", "answer_schema")
    if action == "answered" and not source_ids:
        raise LLMError("Содержательный ответ не указал подтверждающие источники.", "answer_schema")
    if action == "clarification":
        if not question:
            raise LLMError("Уточнение не содержит question.", "answer_schema")
        answer = question
    elif question:
        # Старый контракт question относится к чистому уточнению. Следующий
        # вопрос после ответа должен быть виден и в терминальном answer.
        if "?" not in answer and " ".join(question.casefold().split()) not in " ".join(answer.casefold().split()):
            answer += "\n\n" + question
        question = ""
    if len(answer) > 10000:
        raise LLMError("Слишком длинный ответ.", "answer_schema")
    return {"action": action, "answer": answer, "question": question,
            "source_ids": list(dict.fromkeys(source_ids))}
