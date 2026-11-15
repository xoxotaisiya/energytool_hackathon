"""Разговорные формы → термины базы; факты и ответы сюда не добавляются."""
import re


def bill_problem(text):
    # Сравнение расхода разных цехов не означает рост платежа во времени.
    financial = re.search(r"\bсч[её]т(?:а|е|ов)?\b|плат[её]ж|\bплат(?:им|ить|или|ят)\b|оплат|расход[ыа-яё]* на электр", text, re.I)
    change = re.search(r"вырос|раст[её]т|рост|больше|увелич|подорож|дороже|обман|мошенн|завыш", text, re.I)
    return bool(financial and change)


def price_request(text):
    return bool(re.search(r"цен[аыуе]|стоимост|сколько[^.!?]{0,25}стоит|во сколько[^.!?]{0,35}(?:выйдет|обойд[её]тся)|бюджет", text, re.I))


def usage_comparison(text):
    return bool(re.search(r"цех|участ[а-яё]*|объект|арендатор", text, re.I) and
                re.search(r"сравн|разниц|какой.*больше|где.*больше|видеть.*расход", text, re.I) and
                re.search(r"расход|потреблен|электрич|энерг", text, re.I))


def normalize_query(text):
    if bill_problem(text):
        text += " рост счёта потребление средняя цена плата за мощность сравнить периоды"
    text = re.sub(r"\b(?:симк[а-яё]*|sim(?:[- ]?карт[а-яё]*)?)\b", "сим-карты", text, flags=re.I)
    text = re.sub(r"\bфасад[а-яё]*\b", "фасад вертикальные поверхности зданий", text, flags=re.I)
    text = re.sub(r"\bдвухтариф[а-яё]*\b", "многотарифные раздельно по тарифам", text, flags=re.I)
    text = re.sub(r"\bрасход(?:ы|а)?\s+электричеств[а-яё]*\b", "потребление электроэнергии", text, flags=re.I)
    text = re.sub(r"\bмагазин[а-яё]*\b", "магазины сетевой ретейл", text, flags=re.I)
    text = re.sub(r"\bрасшир[а-яё]*\b", "расширение поэтапный ввод наращивание мощности", text, flags=re.I)
    text = re.sub(r"\bпровер[а-яё]*([^.!?\n]{0,40})уч[её]т[а-яё]*\b", r"аудит существующей системы учёта \1", text, flags=re.I)
    text = re.sub(r"\bпровер[а-яё]*([^.!?\n]{0,20})сч[её]т(?:а|е)?\b", "сравнить показатели счёта", text, flags=re.I)
    text = re.sub(r"\bпонадоб[а-яё]*\b", "понадобятся сведения данные", text, flags=re.I)
    text = re.sub(r"\bоценк[а-яё]*\b", "оценка расчёт", text, flags=re.I)
    return text


PRODUCT_PATTERNS = {
    "solar": r"солнеч|\bсэс\b|панел|инсоляц|сетев[а-яё]*\s+станц|гибридн.*станц",
    "orem": r"оптов|\bорэм\b|прям[а-яё]*\s+(?:участ|(?:самостоятельн[а-яё]*\s+)?выход)",
    "askue": r"платформ|кабинет|цех|по арендат|автоматич.*уч[её]т|автоматиз[а-яё]*\s+(?:сбор|уч[её]т)|сбор показаний автоматически|\bаску[эе]\b",
    "losses": r"потер[ьи]|хищен|воровств",
    # «Присылают фото» описывает текущий сбор показаний и само по себе
    # не выбирает продукт распознавания вместо уже обсуждаемой платформы.
    "meter-reading": r"распознав|по фото|фотограф",
}
PRODUCTS = frozenset(PRODUCT_PATTERNS)


def dialog_products(history):
    """Явная новая тема заменяет прежнюю, а короткое продолжение наследует её."""
    for item in reversed(history):
        if item["role"] != "user":
            continue
        found = infer_products(item["content"])
        explicit = [p for p in found if p in PRODUCTS]
        if explicit:
            # Запрос о фотографиях явно выбирает распознавание; описание
            # ручного сбора фото само по себе не является этим запросом.
            if "meter-reading" in explicit and re.search(r"распознав|по фото|с фотограф", item["content"], re.I):
                return ["meter-reading"]
            return explicit
        if "energy-basics" in found:
            return ["energy-basics"]
    return []


def infer_products(text):
    products = [p for p, pattern in PRODUCT_PATTERNS.items() if re.search(pattern, text, re.I)]
    if re.search(r"ценов.*категор|реактивн|тариф|сч[её]т.*(?:вырос|рост)|расход.*вырос", text, re.I):
        products.append("energy-basics")
    return products


def question_purpose(text):
    if re.search(r"с чего[^.!?\n]{0,25}нач|что[^.!?\n]{0,30}(?:нужно учесть|подготовить|собрать)|что[^.!?\n]{0,35}(?:понадоб|потребует)[^.!?\n]{0,30}оцен|какие[^.!?\n]{0,30}(?:данные|сведения|информац|факторы)|(?:таких|этих) данных[^.!?\n]{0,30}(?:хват|достаточ)", text, re.I):
        return "inputs"
    return "fact"
