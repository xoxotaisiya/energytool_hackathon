"""Связи величин и условий: значения читаются из пользователя и исходных цитат."""
import re
from llm.qwen import LLMError

POWER = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(кВт|МВт)\b(?!\s*(?:[·⋅*]\s*)?[чh])", re.I)


def power_value(text):
    match = POWER.search(text)
    if not match:
        return None
    return {"number": float(match[1].replace(",", ".")), "unit": match[2],
            "kw": float(match[1].replace(",", ".")) * (1000 if match[2].lower() == "мвт" else 1),
            "label": match[0]}


def market_bounds(units):
    result = {}
    for unit in units:
        if unit.get("product") != "orem" or unit.get("role") == "instruction":
            continue
        text = unit["quote"]
        measures = list(POWER.finditer(text))
        if len(measures) >= 2 and re.search(r"от\s*$", text[:measures[0].start()], re.I) and re.search(r"менее", text[measures[0].end():measures[1].start()], re.I):
            if re.search(r"посредник|энергосбытов", text, re.I):
                result["range"] = {"lower": power_value(measures[0][0]), "upper": power_value(measures[1][0]),
                                   "lower_inclusive": True, "upper_inclusive": False, "proof": unit}
        elif len(measures) == 1 and re.search(r"от\s*$", text[:measures[0].start()], re.I) and re.search(r"возмож.*прям|прям.*возмож", text, re.I):
            result["direct"] = {"lower": power_value(measures[0][0]), "proof": unit}
        elif len(measures) == 1 and re.search(r"ниже\s*$", text[:measures[0].start()], re.I) and re.search(r"невозмож|исключает|не допуска", text, re.I):
            result["minimum"] = {"lower": power_value(measures[0][0]), "proof": unit}
    return result


def market_answer(units, facts):
    bounds = market_bounds(units)
    value = power_value(facts.get("connected_power", {}).get("quote", ""))
    if not value:
        return {"missing": "connected_power", "bounds": bounds}
    if "minimum" in bounds and value["kw"] < bounds["minimum"]["lower"]["kw"]:
        rule = bounds["minimum"]
        answer = f"Указанная присоединённая мощность {value['label']} ниже {rule['lower']['label']}. По условиям базы выход на оптовый рынок для этого случая невозможен."
        mode = "below_minimum"
    elif "range" in bounds and bounds["range"]["lower"]["kw"] <= value["kw"] < bounds["range"]["upper"]["kw"]:
        rule = bounds["range"]
        answer = (f"Присоединённая мощность предприятия {value['label']} попадает в диапазон от {rule['lower']['label']}, но менее {rule['upper']['label']}. "
                  "По базе в этом диапазоне участие возможно через энергосбытовую компанию-посредника, а не напрямую. "
                  "Соответствие порогу не гарантирует пригодность проекта: нужна проверка учёта и расчёт по объекту.")
        mode = "intermediary"
    elif "direct" in bounds and value["kw"] >= bounds["direct"]["lower"]["kw"]:
        rule = bounds["direct"]
        answer = (f"По указанной присоединённой мощности {value['label']} достигнут описанный в базе порог {rule['lower']['label']}: прямое участие возможно. "
                  "Это возможность по порогу мощности, а не гарантия участия: остаются требования к учёту и процедурам.")
        mode = "direct_possible"
    else:
        return None
    return {"answer": answer, "mode": mode, "value": {**value, "entity": "connected_power"},
            "proof": rule["proof"], "bounds": bounds}


def validate_relations(text, evidence):
    source = " ".join(u["quote"] for u in evidence)
    if any(u.get("product") == "askue" for u in evidence) and re.search(r"в год|/год|годов", source, re.I):
        if re.search(r"стоимость (?:самого )?прибора|цена (?:самого )?прибора|прибор[^.!?]{0,20}стоит", text, re.I) and not re.search(r"услуг|платформ", text, re.I):
            raise LLMError("Годовая стоимость платформенной услуги за прибор не является ценой самого прибора.", "answer_schema")
        if re.search(r"\d[^.!?]{0,20}(?:руб|₽)", text, re.I):
            if not re.search(r"в год|/год|годов", text, re.I) or not re.search(r"прибор|сч[её]тчик", text, re.I):
                raise LLMError("В цене услуги нужно сохранить период и единицу тарификации: за прибор в год.", "answer_schema")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if "ISO" in sentence and re.search(r"проект[а-яё]*[^.!?]{0,45}(?:соответств|сертифиц)|(?:соответств|сертифиц)[^.!?]{0,45}проект", sentence, re.I):
            if re.search(r"производств.*ISO", source, re.I):
                raise LLMError("Свойство производства оборудования по ISO нельзя переносить на весь проект.", "answer_schema")
        if re.search(r"всех типов|любых типов|всем типам|все конфигурац", sentence, re.I) and re.search(r"сетев.*станц", source, re.I):
            raise LLMError("Ограничение сетевой конфигурации не относится автоматически ко всем типам установок.", "answer_schema")
    # Даже все правильные цифры не разрешают потерять исключённую верхнюю границу.
    bounds = market_bounds(evidence)
    if "range" in bounds and re.search(r"прям", text, re.I) and re.search(r"не предусмотр|невозмож|не может", text, re.I):
        if re.search(r"и выше|и более", text, re.I) and not re.search(r"менее|ниже", text, re.I):
            raise LLMError("Запрет прямого участия потерял верхнюю границу диапазона своего источника.", "answer_schema")
