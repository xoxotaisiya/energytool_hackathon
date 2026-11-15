"""Проверяемые сведения клиента. Источник — только пользовательские реплики."""
import re
from consultant.relations import power_value, POWER

SLOT_QUESTIONS = {
    "object_type": "О каком объекте идёт речь?",
    "region": "В каком регионе находится объект?",
    "roof_age": "Сколько лет кровле?",
    "roof_orientation": "Как ориентированы скаты крыши?",
    "connected_power": "Какая присоединённая мощность указана в документах?",
    "load_period": "В какое время суток приходится основное потребление электроэнергии?",
    "meter_models": "Какие модели счётчиков установлены?",
    "meter_count": "Сколько приборов учёта нужно подключить?",
    "collection_method": "Как сейчас собираются показания счётчиков?",
    "day_night": "Есть ли в суточных данных отдельные значения потребления за день и ночь?",
    "archive_depth": "За какой период накоплены данные учёта?",
    "price_category": "Какая у вас ценовая категория?",
}
# Связь вопроса с описанием необходимых данных в материалах. Это не новые
# критерии допуска: вопрос разрешён лишь при наличии соответствующего источника.
SLOT_SOURCE_PATTERNS = {
    "object_type": r"тип организац|тип объект|вид.*организац",
    "region": r"регион|географ",
    "roof_age": r"кровл|шифер",
    "roof_orientation": r"ориентац.*скат|скат.*ориентац",
    "connected_power": r"присоедин[её]нн.*мощност",
    "load_period": r"профил.*потреблен|дневн.*нагруз|ноч.*нагруз|режим работы",
    "meter_models": r"модел.*прибор|модел.*сч[её]тчик|тип прибор",
    "meter_count": r"количеств.*прибор|числ.*прибор",
    "collection_method": r"сбор[а-яё]* данных|сбор[а-яё]* показан|схем.*сбор",
    "day_night": r"разбивк.*день.*ноч|дневн.*ночн",
    "archive_depth": r"архив|глубин.*год|не менее одного года",
    "price_category": r"ценов.*категор",
}


def customer_facts(history):
    """Сохраняет цитаты, без вывода отсутствующих значений и без слов ассистента."""
    facts = {}
    def put(slot, match, turn):
        if match:
            facts[slot] = {"value": match.group(0).strip(), "quote": match.group(0).strip(), "turn": turn}

    for turn, item in enumerate(history):
        if item.get("role") != "user":
            continue
        text = item.get("content", "")
        low = text.lower().replace("ё", "е")
        patterns = {
            "object_type": r"электросетевая компания|сетев[а-я]+ организац[а-я]+|бизнес[ -]центр[а-я]*|производств[а-я]*|завод[а-я]*|склад[а-я]*|магазин[а-я]*|частн[а-я]+ (?:жил[а-я]+ )?дом[а-я]*|\bснт\b",
            "roof_age": r"(?:кровл[а-я]*|крыш[а-я]*|шифер[а-я]*)[^.!?\n]{0,25}\d+[ -]*(?:лет|год[а-я]*)|(?:стар[а-я]+ )?(?:шиферн[а-я]+ )?(?:кровл[а-я]*|крыш[а-я]*)[^.!?\n]{0,12}стар[а-я]+",
            "roof_material": r"шифер[а-я]*|металлочерепиц[а-я]*|мягк[а-я]+ кровл[а-я]*",
            "roof_orientation": r"(?:восток[а-я]*[^.!?\n]{0,18}запад[а-я]*|запад[а-я]*[^.!?\n]{0,18}восток[а-я]*|север[а-я]*[^.!?\n]{0,18}юг[а-я]*|южн[а-я]+ скат[а-я]*)",
            "meter_models": r"(?:меркурий|энергомера|нева|милур)[ -]*[\w.-]+",
            "meter_count": r"(?:\d+|сорок[а-я]*|двадцат[а-я]*|десят[а-я]*)[ -]*(?:сч[её]тчик[а-я]*|прибор[а-я]*)",
            "collection_method": r"[^.!?\n]{0,45}(?:присыла[а-я]* фото|вручную|ходит техник|снима[а-я]* техник|автоматическ[а-я]+ сбор)[^.!?\n]{0,30}",
            "archive_depth": r"(?:архив[^.!?\n]{0,30}|за\s+)(?:\d+|два|двух|один|одного|три|трех)[ -]*(?:год[а-я]*|лет|месяц[а-я]*)",
            "price_category": r"(?:ценов[а-я]+ категори[а-я]*[^.!?\n]{0,12}(?:[1-6]|перв[а-я]*|втор[а-я]*|трет[а-я]*|четв[её]рт[а-яё]*|пят[а-я]*|шест[а-я]*)|(?:перв[а-я]*|втор[а-я]*|трет[а-я]*|четв[её]рт[а-яё]*|пят[а-я]*|шест[а-я]*)[ -]+(?:ценов[а-я]+ )?категори[а-я]*)",
            "region": r"(?:в|из|регион\s*[:—-]?)\s+(?:[А-ЯЁ][а-яё-]+(?:\s+[а-яё-]+)?\s+(?:област[ьи]|кра[ейя])|Москв[а-я]*|Владивосток[а-я]*|Санкт-Петербург[а-я]*)",
        }
        for slot, pattern in patterns.items():
            put(slot, re.search(pattern, text, re.I if slot != "region" else 0), turn)
        if re.search(r"архив[а-яё]*\s+(?:нет|отсутств)|(?:нет|без)\s+архив", text, re.I):
            facts["archive_depth"] = {"value": "missing", "status": "missing", "quote": text.strip(), "turn": turn}
        elif re.search(r"(?:не знаю|неизвест[а-яё]*)[^.!?]{0,20}(?:глубин|период)[^.!?]{0,15}архив|(?:глубин|период)[^.!?]{0,15}архив[^.!?]{0,15}(?:не знаю|неизвест|уточню)", text, re.I):
            facts["archive_depth"] = {"value": "unknown", "status": "unknown", "quote": text.strip(), "turn": turn}
        # Установленная мощность солнечной станции и присоединённая мощность
        # предприятия — разные сведения. Общее «60 кВт» не заполняет этот слот.
        put("connected_power", re.search(r"присоедин[её]нн[а-я]*\s+мощност[а-я]*[^.!?\n]{0,25}\d+(?:[.,]\d+)?\s*(?:квт|мвт)", text, re.I), turn)
        # Каждая мощность связана со своим существительным, а не с первым
        # числом сообщения. Реплика может содержать оба разных значения.
        put("project_power", re.search(r"(?:солнечн[а-яё]*\s+)?станци[а-яё]*[^.!?\n,;]{0,24}\d+(?:[.,]\d+)?\s*(?:квт|мвт)\b", text, re.I), turn)
        for slot, subject in (("connected_power", r"присоедин[её]нн.*мощност"),
                              ("project_power", r"мощност.*станци|станци")):
            # Явное «не X, а Y» исправляет значение того же объекта.
            correction = re.search(r"\bне\s+(\d+(?:[.,]\d+)?\s*(?:кВт|МВт))\s*[,;]?\s*а\s+(\d+(?:[.,]\d+)?\s*(?:кВт|МВт))\b", text, re.I)
            if correction and (re.search(subject, text[:correction.start()], re.I) or
                               slot in facts and re.search(r"уточн|исправ|ошиб", text, re.I)
                               and not re.search(r"станци|присоедин", text, re.I)):
                facts[slot] = {"value": correction[2], "quote": correction[2], "turn": turn,
                               "correction_quote": correction[0]}
            if re.search(subject, text, re.I) and re.search(r"не знаю|неизвест|уточню", text, re.I) and not POWER.search(text):
                facts[slot] = {"value": "unknown", "status": "unknown", "quote": text.strip(), "turn": turn}
        if turn and re.search(r"присоедин[её]нн.*мощност", history[turn-1].get("content", ""), re.I):
            if re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*(?:кВт|МВт)[.! ]*", text, re.I):
                facts["connected_power"] = {"value": text.strip(), "quote": text.strip(), "turn": turn}
        for slot in ("connected_power", "project_power"):
            fact = facts.get(slot)
            if fact and fact["turn"] == turn:
                value = power_value(fact["quote"])
                # Проверяем единицу в исходной реплике, где ещё не потерян суффикс ·ч.
                if value and re.search(re.escape(value["label"]) + r"\s*(?:[·⋅*]\s*)?[чh]", text, re.I):
                    facts.pop(slot, None)
                elif value:
                    fact.update({"quantity": {**value, "entity": slot}})
        cleaned = re.sub(r"\bдобры[йе]\s+(?:вечер|день|утро)\b", "", text, flags=re.I)
        for sentence in re.split(r"[.!?\n;,]", cleaned):
            if (re.search(r"работ|смен|нагруз|потребл|тратим|расход", sentence, re.I)
                    and not re.search(r"разбивк|дневн.*ночн|день\s*[/и]\s*ноч", sentence, re.I)):
                match = re.search(r"вечер[а-яё]*|ноч[а-яё]*|дн[её]м|дневн[а-яё]*", sentence, re.I)
                if match and re.search(r"\bне\s+(?:работ|потребл)|нагрузк[а-яё]*\s+нет", sentence, re.I):
                    facts["excluded_load_period"] = {"value": match[0], "status": "missing", "quote": sentence.strip(), "turn": turn}
                    if facts.get("load_period", {}).get("value") == match[0]:
                        facts.pop("load_period", None)
                else:
                    put("load_period", match, turn)
        previous = history[turn-1].get("content", "") if turn else ""
        day_night = bool((re.search(r"разбивк|раздел|отдельн|раздельн", low) or re.search(r"дневн.*ночн.*(?:данн|значен|показан)", low))
                         and re.search(r"день|днев", low) and "ноч" in low)
        short_reply = bool(re.search(r"день.*ноч|дневн.*ночн", previous, re.I)
                           and re.fullmatch(r"(?:да|есть|нет|не знаю|неизвестно)[.! ]*", low))
        correction = bool("day_night" in facts and re.search(r"разбивк|разделени|раздельн.*данн", low)
                          and re.search(r"есть|нашли|появил|нет|отсутств|не знаю|неизвест", low))
        if day_night or short_reply or correction:
            status = "unknown" if re.search(r"не знаю|неизвест|не уверен|есть ли|проверю|уточню", low) else (
                "missing" if re.search(r"\bнет\b|отсутств|без разбив|не (?:вед|собир|сохр)", low) else "satisfied")
            facts["day_night"] = {"value": status, "status": status, "quote": text.strip(), "turn": turn}
        # В диалоге категорию часто называют ответом на вопрос ассистента.
        if turn and re.search(r"ценов.*категор", history[turn-1].get("content", ""), re.I):
            put("price_category", re.search(r"\b(?:[1-6]|перв[а-я]*|втор[а-я]*|трет[а-я]*|четв[её]рт[а-я]*|пят[а-я]*|шест[а-я]*)\b", text, re.I), turn)
    return facts


def human_request(text):
    """Только намерение связаться, а не любое упоминание специалиста."""
    requested = False
    for sentence in re.split(r"[.!?\n;,]", text.lower()):
        if not re.search(r"специалист|менеджер|оператор|человек", sentence):
            continue
        if re.search(r"не\s+(?:хочу|нужен|нужно|надо|соединя|переводи|связыва)|без специалист", sentence):
            requested = False
            continue
        if re.search(r"(?:хочу|можно|давайте|прошу|нужно)[^.!?]{0,65}(?:поговор|связ|обсуд|звон)|(?:хочу|нужен|позовите)\s+(?:мне\s+)?(?:жив[а-яё]+\s+)?(?:специалист|менеджер|оператор|человек)|соедините|переведите|свяжите|передайте", sentence):
            requested = True
    return requested


def human_declined(text):
    return not human_request(text) and bool(re.search(
        r"не\s+(?:хочу|нужен|нужно|надо|соединя[а-яё]*|переводи[а-яё]*|связыва[а-яё]*)[^.!?;,]{0,55}(?:специалист|менеджер|оператор|человек)", text, re.I))


def cancels_pending_handoff(text):
    """Отмена относится к текущему действию, даже без повторения его имени."""
    return not human_request(text) and (human_declined(text) or bool(re.search(
        r"передумал|отмен[а-яё]*|пока не (?:надо|нужно)|продолжим (?:здесь|в чате)|давайте.*(?:здесь|в чате)|звонить не надо|без звонка", text, re.I)))


def allowed_questions(units, facts, asked):
    available = {}
    for slot, pattern in SLOT_SOURCE_PATTERNS.items():
        if slot in asked or slot in facts and facts[slot].get("status") != "unknown":
            continue
        sources = [u["id"] for u in units if re.search(pattern, u["quote"] + " " + u["section"], re.I)]
        if sources:
            available[slot] = {"question": SLOT_QUESTIONS[slot], "evidence_ids": sources}
    return available
