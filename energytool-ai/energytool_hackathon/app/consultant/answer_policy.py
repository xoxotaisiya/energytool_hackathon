"""Контракт ответа: цель вопроса, применимость цитат и проверяемые условия.

Не содержит ответов demo или их ID. Предметные требования читаются из
найденных цитат; код управляет полнотой и состоянием диалога.
"""
import re
from llm.qwen import LLMError
from rag.query_normalization import bill_problem, dialog_products, price_request, usage_comparison, question_purpose, PRODUCTS
from rag.retriever import BM25Index


def task_for(question, history):
    user_text = " ".join(h["content"] for h in history if h["role"] == "user")
    products = dialog_products(history)
    if re.search(r"[«\"].*(?:игнорируй|забуд|придумай|назови любую).*?[»\"]", question, re.I):
        return "quoted_instruction"
    if usage_comparison(question):
        return "usage_comparison"
    if bill_problem(question):
        return "bill_diagnosis"
    if price_request(question) and not set(products).intersection(PRODUCTS) and not re.search(r"ветр|гидро|атом|смартфон|интернет", question, re.I):
        return "ambiguous_station" if re.search(r"станци", question, re.I) else "unspecified_price"
    if price_request(question) and set(products).intersection(PRODUCTS):
        return "product_price"
    if not products and re.search(r"выбрать|подобрать|какое решение|какой продукт", question, re.I):
        return "product_selection"
    if re.search(r"окуп|экономическ.*смысл", question, re.I):
        if re.search(r"станци", question, re.I) and "solar" not in products:
            return "ambiguous_station"
        if "solar" in products:
            return "solar_economics"
    if "losses" in products and re.search(r"хват|достаточ|подойдут|можно.*нач|позвол.*нач|минимальн.*(?:требован|услов)|разбивк|архив", question, re.I):
        return "data_sufficiency"
    if "orem" in products and (re.search(r"подход|доступ|достаточ|можно|можем|гарант|возмож|порог|мощност", question, re.I)
                               or re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*(?:кВт|МВт)[.! ]*", question, re.I)) and not price_request(question):
        return "market_eligibility"
    if question_purpose(question) == "inputs":
        return "preparation"
    if set(products).intersection(PRODUCTS) and re.search(r"^(?:мы\s+)?(?:хотим|планируем|нам нужна?|нужна?)\b", question, re.I) and "?" not in question:
        return "product_advice"
    return "product_question"


def applicable(unit, question, task, facts=None):
    if unit.get("role") == "instruction":
        return False
    context = unit["section"] + " " + unit["quote"]
    if unit.get("role") == "constraint":
        known = question + " " + " ".join(v["quote"] for v in (facts or {}).values())
        if re.search(r"частн.*жил.*дом", unit["quote"], re.I) and not re.search(r"частн.*дом|жил.*дом", known, re.I):
            return False
        if re.search(r"преимущественно ночью", unit["quote"], re.I) and not re.search(r"ноч|вечер", known, re.I):
            return False
    if re.search(r"сосед|другого предприятия", context, re.I) and not re.search(r"сосед|друг.*предприят", question, re.I):
        return False
    if re.search(r"цех", question, re.I) and re.search(r"ретейл|ритейл|общепит|сетевой бизнес", context, re.I) and "цех" not in unit["quote"].lower():
        return False
    if task == "solar_economics" and unit["product"] not in {"solar", "general"}:
        return False
    if task == "data_sufficiency" and unit["product"] not in {"losses", "general"}:
        return False
    return True


def select_for_contract(selected, pool, question, task, facts=None):
    selected = [u for u in selected if applicable(u, question, task, facts)]
    pool = [u for u in pool if applicable(u, question, task, facts)]
    mandatory = []
    if task == "bill_diagnosis":
        guidance = [u for u in pool if u.get("kind") == "guidance" and re.search(r"сравн.*период", u["quote"], re.I)]
        scenarios = [u for u in pool if u.get("kind") == "conditional_scenario"]
        # Вечерний график делает профиль релевантной гипотезой, но не диагнозом.
        scenarios.sort(key=lambda u: not (re.search(r"вечер|смен", question, re.I) and "ваш профиль" in u["quote"]))
        category = [u for u in scenarios if re.search(r"ценов[а-яё]* категори", u["quote"], re.I)]
        mandatory = scenarios[:2] + guidance[:1] + category[:1]
        selected = [u for u in selected if u.get("kind") in {"conditional_scenario", "guidance"}]
    elif task == "data_sufficiency":
        mandatory = [u for u in pool if re.search(r"минимально|не менее.*год", u["quote"], re.I)][:3]
    elif task == "solar_economics":
        mandatory = [u for u in pool if re.search(r"в расч[её]те учитываются регион|разброс определяется регионом", u["quote"], re.I)][:2]
    elif "цех" in question.lower():
        direct = [u for u in pool if u["product"] == "askue" and "цех" in u["quote"].lower()]
        if direct:
            mandatory = direct[:2]
            selected = [u for u in selected if u["product"] == "askue"]
    return list({u["id"]: u for u in mandatory + selected}.values())[:12]


def contract_for(task, selected, questions):
    required = []
    facets = []
    if task == "bill_diagnosis":
        required = [u["id"] for u in selected if u.get("kind") == "guidance" and re.search(r"сравн.*период", u["quote"], re.I)][:1]
        required += [u["id"] for u in selected if u.get("kind") == "conditional_scenario"][:1]
        facets = ["Только возможные причины, без установленного диагноза",
                  "Сравнение двух сопоставимых периодов: объём потребления, средняя цена за кВт·ч, плата за мощность"]
    if task == "solar_economics":
        facets = ["Не гарантировать окупаемость; применимость к объекту требует расчёта",
                  "Не подменять вечернюю нагрузку ночной и не переносить чужой числовой пример"]
    return {"task": task, "required_evidence_ids": required, "required_facets": facets,
            "required_question_slot": "price_category" if task == "bill_diagnosis" and "price_category" in questions else ""}


def validate_content(answer, output, contract, question):
    task = contract["task"]
    used = {i for c in output.get("claims", []) for i in c.get("evidence_ids", [])}
    if any(i not in used for i in contract["required_evidence_ids"]):
        raise LLMError("Пропущен обязательный источник способа проверки, хотя он найден.", "answer_schema")
    if re.search(r"сосед", answer, re.I) and not re.search(r"сосед", question, re.I):
        raise LLMError("Ответ подменяет ситуацию клиента сравнением с соседом.", "answer_schema")
    if task == "bill_diagnosis":
        if re.search(r"(?:а|но)\s+не вся сумма|(?:сумма|сч[её]т)[^.!?]{0,16}не (?:раст[её]т|увеличивается)", answer, re.I):
            raise LLMError("Ответ противоречит сообщению клиента о росте общей суммы счёта. Не противопоставляй рост составляющей росту суммы без данных.", "answer_schema")
        if not re.search(r"может|возмож|предполож", answer, re.I):
            raise LLMError("Нужны возможные причины, а не только инструкция сравнения или установленный диагноз.", "answer_schema")
        if re.search(r"снижен.*отрасл|цена верн[её]тся|смена поставщика", answer, re.I) and not re.search(r"смен.*поставщик|прогноз|верн[её]тся|снижени.*цен", question, re.I):
            raise LLMError("Ответ добавляет непрошеные выводы о смене поставщика или прогнозе цен.", "answer_schema")
        checks = [r"сравн|сопостав", r"период", r"объ[её]м|потреблен|расход", r"средн.*цен|цен.*кВт|цен.*киловатт", r"плат.*мощност"]
        if not all(re.search(pattern, answer, re.I) for pattern in checks):
            raise LLMError("Неполный разбор роста счёта: нужны сравнение периодов, потребление, средняя цена и плата за мощность.", "answer_schema")
        if re.search(r"обман", question, re.I) and not re.search(r"нельзя|недостаточ|не доказ|не означ|не подтвержд|без.*данн", answer, re.I):
            raise LLMError("Рост счёта сам по себе не доказывает обман; обозначь неопределённость.", "answer_schema")
    if task == "solar_economics" and not re.search(r"расч[её]т|нельзя.*гарант|зависит|недостаточ", answer, re.I):
        raise LLMError("Окупаемость объекта требует расчёта, а не безусловного обещания.", "answer_schema")


def safe_quotes(selected, constraints, question, task, facts=None):
    """Позиция selected[0] не является доказательством применимости."""
    candidates = [u for u in selected if applicable(u, question, task, facts)]
    if task == "solar_economics":
        candidates = [u for u in candidates if re.search(r"окуп|эконом|расч[её]т|выработ.*нагруз|профил.*потреблен", u["quote"], re.I)]
    if not candidates:
        return []
    if task == "bill_diagnosis":
        scenarios = [u for u in candidates if u.get("kind") == "conditional_scenario"][:2]
        guidance = [u for u in candidates if u.get("kind") == "guidance"][:1]
        return list({u["id"]: u for u in scenarios + guidance + constraints}.values())
    if task == "solar_economics":
        calculation = [u for u in candidates if re.search(r"в расч[её]те учитываются регион|разброс определяется регионом", u["quote"], re.I)]
        if calculation:
            return list({u["id"]: u for u in calculation[:1] + constraints}.values())
    index = BM25Index([{"path": u["section"], "text": u["quote"]} for u in candidates])
    scores = dict(index.search(question, len(candidates)))
    ranked = sorted(enumerate(candidates), key=lambda row: (-scores.get(row[0], 0), row[0]))
    lead = next((u for i, u in ranked if scores.get(i, 0) > 0 and u.get("role") != "example"), None)
    if lead is None:
        return []
    return list({u["id"]: u for u in [lead] + constraints if applicable(u, question, task, facts)}.values())[:3]


def render_quote(unit):
    """Клиентская форма исходного блока без служебных колонок таблицы."""
    quote = unit["quote"].lstrip("- ")
    if unit.get("guard"):
        return "Ограничение применимости относится к ситуации: «" + quote + "»"
    if unit.get("kind") == "conditional_scenario":
        fields = dict(part.split(": ", 1) for part in quote.split("; ") if ": " in part)
        subject = fields.get("Что выросло")
        sign = fields.get("Как это выглядит в счёте", fields.get("Как это выглядит"))
        if subject and sign:
            return "Возможный сценарий: " + subject + " — " + sign.rstrip(".") + "."
        return "Возможный сценарий: «" + quote + "»"
    if unit.get("kind") == "guidance" and re.search(r"сравн.*период", quote, re.I):
        return re.split(r"(?<=[.!?])\s+", quote)[0]
    if "Что закрывает платформа: " in quote:
        return "Платформа поддерживает " + quote.split("Что закрывает платформа: ", 1)[1].split("; ", 1)[0].rstrip(".") + "."
    return quote


def minimum_data(selected, facts):
    """Минимальные условия только из найденного требования к данным."""
    proof = [u for u in selected if u["product"] == "losses" and re.search(r"минимально|минимальн.*данн", u["quote"], re.I)]
    if not proof:
        return None
    source = " ".join(u["quote"] for u in proof)
    if not re.search(r"день|дневн", source, re.I) or not re.search(r"ноч", source, re.I):
        return None
    day = facts.get("day_night", {}).get("status", "unknown")
    archive = facts.get("archive_depth", {}).get("quote", "")
    years = re.search(r"(\d+|два|двух|один|одного|три|тр[её]х|полгода)[ -]*(год[а-я]*|лет|месяц[а-я]*)?", archive, re.I)
    archive_status = "missing" if facts.get("archive_depth", {}).get("status") == "missing" else "unknown"
    # Порог не зашит: сравниваем только если годовой минимум присутствует в цитате.
    if archive_status != "missing" and re.search(r"не менее (?:одного|1) года|не менее года", source, re.I) and years:
        word = years[1].lower()
        number = int(word) if word.isdigit() else {"два": 2, "двух": 2, "один": 1, "одного": 1, "три": 3, "трёх": 3, "трех": 3, "полгода": .5}[word]
        if years[2] and years[2].startswith("месяц"):
            number /= 12
        archive_status = "satisfied" if number >= 1 else "missing"
    return {"day_night": day, "archive_depth": archive_status, "source_ids": list(dict.fromkeys(u["source_id"] for u in proof)),
            "evidence": proof}
