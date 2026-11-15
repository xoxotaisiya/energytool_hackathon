"""Факты с неизменяемыми цитатами и принадлежностью к источнику/продукту."""
import re
import json

from llm.qwen import LLMError
from rag.query_normalization import PRODUCTS, normalize_query, infer_products, question_purpose
from rag.retriever import BM25Index
from consultant.relations import validate_relations


def prompt_evidence(units):
    names = {"solar": "Солнечные электростанции", "askue": "Облачная платформа учёта",
             "orem": "Выход на оптовый рынок", "losses": "Поиск коммерческих потерь",
             "meter-reading": "Распознавание показаний по фотографии", "energy-basics": "Справочные сведения об энергетике"}
    return [{"id": u["id"], "product": names.get(u["product"], u["section"].split(" › ")[0]), "section": u["section"],
             "section_type": u.get("section_type", ""), "role": u.get("role", "fact"),
             "kind": u.get("kind", "fact"),
             "quote": u["quote"]} for u in units]


def bounded_evidence(units, question, max_chars):
    """Отбор целых фактов внутри найденных чанков, без обрезки условий."""
    if not units:
        return []
    query = normalize_query(question)
    active = infer_products(question)
    purpose = question_purpose(question)
    index = BM25Index([{"path": u["section"], "text": u["quote"]} for u in units], prefix=5, path_weight=1)
    scores = dict(index.search(query, len(units)))
    ranked = sorted(enumerate(units), key=lambda row: (
        -int(purpose == "inputs" and row[1].get("section_type") == "inputs-needed"),
        -scores.get(row[0], 0), -int(row[1]["product"] in active), row[0]))
    kept, used = [], 2
    for _, unit in ranked:
        cost = len(json.dumps(prompt_evidence([unit])[0], ensure_ascii=False)) + 2
        if used + cost <= max_chars:
            kept.append(dict(unit, id=f"F{len(kept)+1}"))
            used += cost
    return kept


def evidence_units(chunks, active_products=(), question=""):
    """Делим лишь по границам абзацев/пунктов, не по точкам и не по длине.

    Полная строка таблицы содержит обе конфигурации. Условия, введение и
    хвост пункта остаются вместе; нормализуются только пробелы.
    """
    result, seen = [], set()
    for chunk in chunks:
        if active_products and chunk.get("product") in (PRODUCTS | {"energy-basics"}) and chunk["product"] not in active_products:
            continue
        # «Автономные и изолированные объекты» — отдельный сценарий. Упоминание
        # сети в сравнении стоимости не делает его фактом о сетевой станции.
        if (re.search(r"сетев[а-яё]*\s+станц", question, re.I)
                and not re.search(r"автоном|изолиров|дизел", question, re.I)
                and re.search(r"автоном|изолиров", chunk.get("path", ""), re.I)):
            continue
        # build_chunks.pack соединяет самостоятельные Markdown-абзацы,
        # пункты и целые строки таблиц переводом строки. Это уже границы
        # исходных блоков, а не переносы по ширине страницы.
        blocks = [line.strip() for line in chunk.get("text", "").splitlines() if line.strip()]
        for block in blocks:
            quote = " ".join(block.split())
            if not chunk.get("guard") and question and unasked_commercial_fact(quote, chunk.get("path", ""), question):
                continue
            key = (chunk["id"], quote)
            if key in seen or not quote:
                continue
            seen.add(key)
            role = "instruction" if (chunk.get("section_type") in {"forbidden", "we-dont"}
                    or re.search(r"чего не стоит ждать|чего утверждать нельзя", chunk.get("path", ""), re.I)) else (
                        "constraint" if chunk.get("section_type") == "not-for" else (
                        "example" if re.search(r"пример", chunk.get("path", ""), re.I) or quote.startswith("Пример:") else "fact"))
            result.append({"id": f"F{len(result)+1}", "source_id": chunk["id"],
                           "role": role,
                           "product": chunk.get("product", ""), "section": chunk.get("path", ""),
                           "section_type": chunk.get("section_type", ""),
                           "kind": "conditional_scenario" if quote.startswith("Что выросло:") else (
                               "guidance" if re.search(r"^Первое, что|^Для начала|^Начать стоит", quote) else "fact"),
                           "quote": quote, "guard": bool(chunk.get("guard"))})
    return result


def unasked_commercial_fact(quote, section, question):
    """Не передаём соседние коммерческие показатели для простого вопроса.

    Ограничения применимости и входные данные не фильтруются по числам.
    Это отбор контекста по теме вопроса, а не новые условия продукта.
    """
    q = question.lower().replace("ё", "е")
    body = quote.lower().replace("ё", "е")
    leaf = section.split(" › ")[-1].lower().replace("ё", "е")
    if (re.search(r"срок.*реализ|сколько времени.*(?:проект|станц)|от подписания.*запуск", leaf + " " + body)
            and not re.search(r"срок|сколько времени|как быстро|долго|месяц|недел|когда.*(?:готов|запуск)", q)):
        return True
    if (re.search(r"руб(?:л[а-я]*)?\b|стоимост|ценообразован", body)
            and not re.search(r"цен|стоимост|сколько.*(?:стоит|денег)|затрат|бюджет|эконом|дорож|дешев|выгод", q)):
        return True
    if "окупаем" in body and not re.search(r"окуп|эконом|выгод|смысл", q):
        return True
    if (re.search(r"гаранти.*\d|ресурс.*\d.*лет", body)
            and not re.search(r"гарант|служб|ресурс|лет|шифер|кровл", q)):
        return True
    if body.startswith("требуемая площадь") and not re.search(r"площад|мест|габарит|метр|кровл|крыш|фасад", q):
        return True
    return False


def select_evidence(ids, units):
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(ids) > 10:
        raise LLMError("Некорректный список evidence_ids.", "evidence_schema")
    lookup = {u["id"]: u for u in units}
    if any(i not in lookup for i in ids):
        raise LLMError("Указан факт вне переданных материалов.", "evidence_schema")
    return [lookup[i] for i in dict.fromkeys(ids)]


def contextual_constraints(units, customer, question=""):
    """Присоединяем ограничения из источника, относящиеся к известной ситуации."""
    selected = []
    late_load = re.search(r"вечер|ноч", customer.get("load_period", {}).get("quote", ""), re.I)
    slate = re.search(r"шифер", customer.get("roof_material", {}).get("quote", ""), re.I)
    if question and not re.search(r"смысл|эконом|нагруз|выработ|подход|резерв", question, re.I):
        late_load = None
    if question and not re.search(r"шифер|кровл|крыш|монтаж|установ|рассматр|подход", question, re.I):
        slate = None
    for unit in units:
        if not unit["guard"]:
            continue
        quote = unit["quote"]
        if (late_load and re.search(r"ноч.*(?:выработ|нагруз)|выработ.*не совпад", quote, re.I)
                or slate and re.search(r"шифер", quote, re.I)):
            selected.append(unit)
    return selected[:2]


def numbers(text):
    return set(re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", ".")))


def quantities(text):
    aliases = {"квт": "power_kw", "мвт": "power_mw", "руб": "rub", "рублей": "rub", "рубля": "rub",
               "рубль": "rub", "%": "percent", "процентов": "percent", "процента": "percent", "процент": "percent",
               "месяц": "month", "месяца": "month", "месяцев": "month", "мес": "month", "лет": "year", "год": "year", "года": "year"}
    found = re.findall(r"(\d+(?:[.,]\d+)?)\s*(кВт\s*[·⋅*]?\s*ч|МВт\s*[·⋅*]?\s*ч|кВт|МВт|рублей|рубля|рубль|руб|%|процентов|процента|процент|месяцев|месяца|месяц|мес|лет|года|год)(?!\w)", text, re.I)
    result = set()
    for value, unit in found:
        compact = re.sub(r"[\s·⋅*]", "", unit.lower())
        kind = {"квтч": "energy_kwh", "мвтч": "energy_mwh"}.get(compact) or aliases[compact]
        result.add((value.replace(",", "."), kind))
    return result


def check_conditions(text, evidence, customer=None):
    customer = customer or {}
    for fact in evidence:
        quote = fact["quote"]
        # Конкретный проект в источнике не становится расчётом для другой
        # мощности. Диапазоны «от ... до ...» остаются диапазонами.
        powers = re.findall(r"(\d+(?:[.,]\d+)?)\s*(кВт|МВт)\b", quote, re.I)
        scoped = powers and not re.search(r"от\s+\d+.*до\s+\d+|\d+\s*[-–—]\s*\d+\s*(?:квт|мвт)", quote, re.I)
        if scoped and re.search(r"месяц|срок|окупаем|рубл|руб\b", text, re.I) and numbers(text):
            clauses = re.split(r"(?<=[.!?])\s+|[,;]\s*(?:а|но)\s+", text)
            for clause in clauses:
                if not re.search(r"месяц|срок|окупаем|рубл|руб\b", clause, re.I) or not numbers(clause):
                    continue
                if re.search(r"неизвест|отдельн|индивидуальн|нельзя|не следует|после.*(?:оцен|расч)", clause, re.I):
                    continue
                for value, unit in powers:
                    if not re.search(re.escape(value) + r"\s*" + re.escape(unit) + r"\b", clause, re.I):
                        raise LLMError("Числовой пример потерял условие мощности своего источника.", "answer_schema")
        if "экономически сомнительно" in quote.lower() and re.search(r"технически невозможно|монтаж запрещ[её]н|нельзя установ", text, re.I):
            raise LLMError("Экономическая оговорка превращена в технический запрет.", "answer_schema")
        load = customer.get("load_period", {}).get("quote", "")
        if re.search(r"вечер", load, re.I) and not re.search(r"ноч", load, re.I) and re.search(r"ноч", quote, re.I):
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                for clause in re.split(r",?\s+(?:поэтому|следовательно|так что)\s+", sentence, flags=re.I):
                    if (re.search(r"выработ.*не совпад|экономик.*не обеспеч", clause, re.I)
                            and not re.search(r"если|может.*не совпад|зависит|нельзя.*(?:оцен|утвержд)", clause, re.I)):
                        raise LLMError("Без расчёта нельзя утверждать несовпадение выработки и вечерней нагрузки. Оговорка об отсутствии гарантии не доказывает эту причину.", "answer_schema")
                if (re.search(r"экономик.*не обеспеч|выработ.*не совпад|(?:проект|станци).*невыгод|не окупит", sentence, re.I)
                        and not re.search(r"если|может|зависит|нельзя.*(?:оцен|утвержд)|отдельн.*расч[её]т", sentence, re.I)):
                    raise LLMError("Ночное ограничение не устанавливает невыгодность или несовпадение выработки для вечернего графика. Нужен расчёт, сохрани условие.", "answer_schema")
                if (re.search(r"выработ|энерги[ия] станции", sentence, re.I)
                        and re.search(r"недостаточ|не хватит|не сможет покрыть", sentence, re.I)
                        and not re.search(r"может|зависит|нельзя.*(?:оцен|утвержд)|неизвест|без.*расч[её]т|если", sentence, re.I)):
                    raise LLMError("По вечернему графику нельзя установить достаточность выработки без расчёта и профиля нагрузки. Ночное ограничение не доказывает этот вывод.", "answer_schema")
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if (re.search(r"ноч|не подходит|неоправдан|бессмыслен|оптимальн", sentence, re.I)
                        and not re.search(r"если|может|зависит|для ночн|при ночн|сопостав|отдельн.*(?:оцен|расч)", sentence, re.I)):
                    raise LLMError("Вечерняя нагрузка не доказывает, что клиент работает преимущественно ночью. Сохрани условие в каждом утверждении.", "answer_schema")


def grounded_claims(raw, selected, customer, available_source_ids):
    """Проверяем атрибуцию и новые цифры. Это не семантический LLM-судья."""
    claims = raw.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 4:
        raise LLMError("Нужны 1–4 утверждения с подтверждениями.", "answer_schema")
    texts, source_ids = [], []
    user_numbers = numbers(" ".join(v["quote"] for v in customer.values()))
    user_quantities = quantities(" ".join(v["quote"] for v in customer.values()))
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            raise LLMError("Некорректное утверждение.", "answer_schema")
        text = claim["text"].strip()
        try:
            evidence = select_evidence(claim.get("evidence_ids"), selected)
        except LLMError:
            raise LLMError("Утверждение ссылается на невыбранный факт.", "answer_schema") from None
        if not evidence or not text or len(text) > 1200 or "?" in text:
            raise LLMError("Утверждение должно быть кратким фактом без вопроса.", "answer_schema")
        # F1 — идентификатор, а не неподтверждённое число «1»: причина
        # отклонения должна сообщать модели, что именно исправить.
        for technical_id in list(available_source_ids) + [u["id"] for u in selected]:
            if re.search(r"(?<![\w-])" + re.escape(technical_id) + r"(?![\w-])", text):
                raise LLMError("В ответ попал внутренний ID. Удали ID из текста утверждения.", "answer_schema")
        if re.search(r"\b(?:askue|orem|meter-reading|energy-basics)\b", text, re.I):
            raise LLMError("В ответ попало внутреннее обозначение продукта.", "answer_schema")
        if any(e.get("role") == "instruction" for e in evidence):
            raise LLMError("Внутренняя инструкция не является фактом для ответа клиенту.", "answer_schema")
        if re.search(r"подтвержден[оа].*раздел|в разделе [«\"]|обещать.*нельзя|нижняя граница.*формулировк|в документации указано", text, re.I):
            raise LLMError("Убери служебные инструкции и ссылки на разделы из клиентского ответа.", "answer_schema")
        if numbers(text) - numbers(" ".join(e["quote"] for e in evidence)) - user_numbers:
            raise LLMError("Утверждение содержит неподтверждённое число.", "answer_schema")
        if quantities(text) - quantities(" ".join(e["quote"] for e in evidence)) - user_quantities:
            raise LLMError("Число перенесено на другую единицу или показатель.", "answer_schema")
        if re.search(r"\b(?:уточните|подскажите|пришлите|укажите|сообщите|скажите|оставьте)\b", text, re.I):
            raise LLMError("Вопрос клиенту должен выбираться через question_slot.", "answer_schema")
        check_conditions(text, evidence, customer)
        validate_relations(text, evidence)
        if any(e.get("kind") == "conditional_scenario" for e in evidence):
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                causal = re.search(r"сч[её]т.*(?:раст|вырос)|рост.*(?:связ|вызв|обусловл)|из-за|причин|у вас.*(?:вырос|увелич|измен)", sentence, re.I)
                uncertain = re.search(r"может|возмож|если|например|\bпри\b|провер|сопостав|сравн|нельзя|недостаточно", sentence, re.I)
                if causal and not uncertain:
                    raise LLMError("Возможный сценарий принят за установленную причину у клиента. Нельзя утверждать причину без сопоставления данных.", "answer_schema")
        texts.append(text)
        source_ids.extend(e["source_id"] for e in evidence)
    return " ".join(texts), list(dict.fromkeys(source_ids))


def validate_input_answer(raw, selected, question):
    if question_purpose(question) != "inputs" or re.search(r"хват|достаточ|подойдут", question, re.I):
        return
    input_ids = {u["id"] for u in selected if u.get("section_type") == "inputs-needed"}
    if not input_ids:
        return
    for claim in raw.get("claims", []):
        if (input_ids.intersection(claim.get("evidence_ids", []))
                and re.search(r"потреб|понадоб|подгот|нач[а-я]* с|переч[её]нь|собра|предостав|нужн.*(?:данн|сведен|модел|тип|схем)", claim.get("text", ""), re.I)):
            return
    raise LLMError("Вопрос о начале работы требует перечислить исходные данные по выбранным источникам, а не просто рекомендовать продукт.", "answer_schema")
