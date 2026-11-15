#!/usr/bin/env python3
"""Разбор базы знаний Energy Tool и нарезка на фрагменты (chunks).

Запуск:
    python build_chunks.py [путь_к_knowledge-base_или_zip] [куда_сохранить_chunks.json]

Что делает:
  1. Читает все .md/.txt из папки (или из knowledge-base.zip, как на платформе).
  2. Разбирает Markdown: заголовки, абзацы, списки, pandoc-таблицы.
  3. Превращает каждую строку таблицы в обычное предложение
     «Показатель: ...; Значение: ...; Оговорка: ...», чтобы цифра и её оговорка
     не разделялись.
  4. Режет текст по смысловым разделам (а не по числу символов), крупные
     разделы делит по абзацам/пунктам/строкам таблицы.
  5. Добавляет метаданные: продукт, тип раздела, флаг «защитный» (guard).
  6. Сохраняет chunks.json и печатает статистику и проверку полноты чисел.

Только стандартная библиотека: пригодится и в solution.py на платформе.
"""
import json
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

MAX_CHARS = 1100    # целевой максимум текста в одном фрагменте
MERGE_TAIL = 250    # слишком короткий хвост склеиваем с предыдущим фрагментом

RULE_RE = re.compile(r"^\s+-{3,}(?:\s+-{3,})*\s*$")   # линия-рамка pandoc-таблицы
HR_RE = re.compile(r"^-{10,}\s*$")                     # горизонтальная черта
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
FILEREF_RE = re.compile(r"(?:\.\./)?(?:[\w-]+/)*(\d\d-[\w-]+\.md)")

LAYERS = {"00-reference": "reference", "10-energy": "energy", "20-products": "product"}

PRODUCT_FILE = {
    "product-map": "overview", "portfolio": "portfolio", "orem": "orem",
    "askue": "askue", "solar": "solar", "losses": "losses",
    "meter-reading": "meter-reading", "general": "general",
}
PRODUCT_HEAD = [
    (r"оптов", "orem"), (r"платформ", "askue"), (r"солнечн", "solar"),
    (r"потер", "losses"), (r"распознав", "meter-reading"), (r"порог", "energy-basics"),
]
STEM_TYPE = {
    "numbers-and-indicators": "numbers", "product-scope-and-limits": "limits",
    "product-map": "overview", "regulation": "regulation",
}
TYPE_RULES = [
    (r"^чего утверждать нельзя", "forbidden"),
    (r"^факты, которые можно утверждать", "facts"),
    (r"^что мы не делаем", "we-dont"),
    (r"^не подходит", "not-for"),
    (r"^(кому подходит|подходит, если)", "for-whom"),
    (r"^что это$", "what"),
    (r"^частые вопросы", "faq"),
    (r"^(какие сведения могут понадобиться|что нужно от клиента|"
     r"для (первичного|анализа|солнечной|платформы|поиска|распознавания))", "inputs-needed"),
    (r"^(быстрая навигация|навигация)", "navigation"),
    (r"^следующий шаг", "next-step"),
    (r"^когда требуется специалист", "handoff"),
    (r"^(бесплатная предварительная оценка|обратный звонок|заявка в рамках|согласие)", "requests"),
]
GUARD_TYPES = {"forbidden", "we-dont", "not-for", "limits"}
PER_ROW_STEMS = {"numbers-and-indicators"}   # там каждая строка таблицы = отдельный факт
SHORT_ID = {"product-map": "map", "numbers-and-indicators": "numbers",
            "product-scope-and-limits": "scope", "calculation-and-requests": "calc",
            "price-categories": "price", "power-thresholds": "power", "meter-reading": "meter"}


# ---------------------------------------------------------------- чтение
def load_docs(src):
    """Возвращает [(относительный_путь, текст)] из папки или zip."""
    src = Path(src)
    docs = []
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for name in sorted(z.namelist()):
                if name.startswith("__MACOSX/") or Path(name).suffix.lower() not in (".md", ".txt"):
                    continue
                docs.append((name, z.read(name).decode("utf-8-sig", errors="replace")))
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".md", ".txt"):
                docs.append((p.relative_to(src).as_posix(), p.read_text(encoding="utf-8-sig")))
    if not docs:
        raise SystemExit(f"В {src} нет файлов .md/.txt")
    return docs


# ---------------------------------------------------------------- очистка
TITLES = {}   # имя_файла.md -> заголовок (для замены ссылок на файлы)


def clean(s):
    """Убирает Markdown-разметку и нормализует текст."""
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)                      # [текст](ссылка) -> текст
    s = FILEREF_RE.sub(lambda m: "«" + TITLES.get(m.group(1), "справочный раздел") + "»", s)
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"(?<![\w*])\*(\S[^*]*?)\*(?![\w*])", r"\1", s)          # *курсив*
    s = s.replace("---", "—").replace("--", "–")
    s = re.sub(r"\\(.)", r"\1", s)                                        # убрать экранирование \#
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- таблицы
def col_spans(rule):
    """Находит границы столбцов в текстовой таблице."""
    return [(m.start(), m.end()) for m in re.finditer(r"-+", rule)]


def split_cells(line, starts):
    """Разделяет строку таблицы на ячейки."""
    cells = []
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < len(starts) else None
        cells.append(line[st:en] if en else line[st:])
    return cells


def parse_table(lines, i):
    """lines[i] — верхняя рамка. Возвращает (блок_таблицы | None, индекс_после)."""
    spans0 = col_spans(lines[i])
    k = i + 1
    header_lines = []
    if len(spans0) > 1:                      # таблица без шапки
        spans = spans0
    else:                                    # шапка между двумя рамками
        while k < len(lines) and not RULE_RE.match(lines[k]):
            k += 1
        if k >= len(lines):
            return None, i + 1
        header_lines = lines[i + 1:k]
        spans = col_spans(lines[k])
        k += 1
    starts = [s for s, _ in spans]
    body = []
    while k < len(lines) and not RULE_RE.match(lines[k]):
        body.append(lines[k])
        k += 1
    end = k + 1
    headers = [""] * len(starts)
    for hl in header_lines:
        for c, cell in enumerate(split_cells(hl, starts)):
            headers[c] = (headers[c] + " " + cell.strip()).strip()
    has_blank = any(not b.strip() for b in body)
    groups, cur = [], []
    for b in body:
        if has_blank:
            if b.strip():
                cur.append(b)
            elif cur:
                groups.append(cur)
                cur = []
        else:
            if b.strip():
                groups.append([b])
    if cur:
        groups.append(cur)
    rows = []
    for g in groups:
        cells = [""] * len(starts)
        for ln in g:
            for c, cell in enumerate(split_cells(ln, starts)):
                cells[c] = (cells[c] + " " + cell.strip()).strip()
        rows.append(cells)
    return ["t", [clean(h) for h in headers], rows], end


def row_text(headers, cells):
    """Собирает строку таблицы в самостоятельное предложение."""
    parts = []
    for h, c in zip(headers, cells):
        c = clean(c)
        if c:
            parts.append(f"{h}: {c}" if h else c)
    t = "; ".join(parts)
    return t if t.endswith(".") else t + "."


# ---------------------------------------------------------------- блоки
def parse_simple_table(lines, i, header_raw):
    """Простая таблица: шапка стоит НАД линией-рамкой, строки идут до пустой строки."""
    starts = [s for s, _ in col_spans(lines[i])]
    headers = [c.strip() for c in split_cells(header_raw, starts)]
    k, rows = i + 1, []
    while k < len(lines) and lines[k].strip() and not RULE_RE.match(lines[k]):
        rows.append([c.strip() for c in split_cells(lines[k], starts)])
        k += 1
    if k < len(lines) and RULE_RE.match(lines[k]):    # необязательная закрывающая рамка
        k += 1
    return ["t", [clean(h) for h in headers], rows], k


def parse_blocks(text):
    """Разбирает Markdown на заголовки, абзацы, списки и таблицы."""
    lines = [re.sub(r"^\\#", "#", ln.rstrip()) for ln in text.replace("\r\n", "\n").split("\n")]
    blocks, para, para_raw = [], [], []
    prev_blank = True
    i = 0

    def flush():
        """Сохраняет накопленный абзац или буфер."""
        if para:
            blocks.append(["p", " ".join(para)])
            para.clear()
            para_raw.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or HR_RE.match(ln):
            flush(); prev_blank = True; i += 1; continue
        if RULE_RE.match(ln):
            if len(col_spans(ln)) > 1 and para:        # шапка над рамкой -> простая таблица
                header_raw = para_raw[-1]
                para.pop(); para_raw.pop()
                flush()
                tbl, i = parse_simple_table(lines, i, header_raw)
            else:
                flush()
                tbl, i = parse_table(lines, i)
            if tbl:
                blocks.append(tbl)
            prev_blank = True
            continue
        m = HEAD_RE.match(ln)
        if m:
            flush(); blocks.append(["h", len(m.group(1)), m.group(2)]); prev_blank = True; i += 1; continue
        it = ITEM_RE.match(ln)
        last_li = bool(blocks) and blocks[-1][0] == "li" and not para
        if it and (prev_blank or last_li):
            marker = re.match(r"^\s*(\d+[.)]|[-*+])", ln).group(1)
            flush(); blocks.append(["li", it.group(2), marker]); prev_blank = False; i += 1; continue
        if last_li and (not prev_blank or len(ln) - len(ln.lstrip()) >= 2):
            blocks[-1][1] += " " + s; prev_blank = False; i += 1; continue
        para.append(s); para_raw.append(ln); prev_blank = False; i += 1
    flush()
    return blocks


def build_sections(blocks, stem):
    """Список разделов: {path: [(уровень, заголовок)...], units: [(вид, текст)...]}."""
    stack, secs = [], []
    cur = {"path": [], "units": []}
    for b in blocks:
        if b[0] == "h":
            if cur["units"]:
                secs.append(cur)
            lvl, title = b[1], clean(b[2])
            stack = [x for x in stack if x[0] < lvl] + [(lvl, title)]
            cur = {"path": list(stack), "units": []}
        elif b[0] == "p":
            cur["units"].append(("p", clean(b[1])))
        elif b[0] == "li":
            marker = b[2] if b[2][0].isdigit() else "-"
            cur["units"].append(("li", marker + " " + clean(b[1])))
        elif b[0] == "t":
            for r in b[2]:
                t = row_text(b[1], r)
                if t.strip(". "):
                    cur["units"].append(("row", t))
    if cur["units"]:
        secs.append(cur)
    return secs


# ---------------------------------------------------------------- нарезка
def split_long(text):
    """Делит длинный текст на части по границам предложений."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > MAX_CHARS:
            out.append(buf); buf = p
        else:
            buf = (buf + " " + p).strip()
    if buf:
        out.append(buf)
    return out


def pack(units, per_row):
    """Объединяет небольшие смысловые блоки в фрагменты для поиска."""
    chunks, buf = [], []

    def size():
        """Считает размер текущего буфера."""
        return sum(len(x) + 1 for x in buf)

    def flush():
        """Сохраняет накопленный абзац или буфер."""
        if buf:
            chunks.append("\n".join(buf)); buf.clear()

    for kind, text in units:
        pieces = split_long(text) if len(text) > MAX_CHARS else [text]
        for piece in pieces:
            if per_row and kind == "row":
                flush(); chunks.append(piece); continue
            if buf and size() + len(piece) > MAX_CHARS:
                flush()
            buf.append(piece)
    flush()
    if not per_row and len(chunks) > 1 and len(chunks[-1]) < MERGE_TAIL:
        tail = chunks.pop()
        chunks[-1] += "\n" + tail
    return chunks


def classify(titles, stem, layer):
    """Определяет тип раздела базы знаний."""
    inner = titles[1:]
    if not inner:
        return STEM_TYPE.get(stem) or ("summary" if layer == "product" else "intro")
    for t in reversed(inner):
        tl = t.lower()
        for rx, typ in TYPE_RULES:
            if re.search(rx, tl):
                return typ
    if stem == "faq" or (stem == "general" and len(titles) >= 3):
        return "faq"
    return STEM_TYPE.get(stem, "content")


def product_of(layer, stem, titles):
    """Определяет продукт, к которому относится раздел."""
    if stem in PRODUCT_FILE:
        return PRODUCT_FILE[stem]
    if layer == "energy":
        return "energy-basics"
    heads = " ".join(titles[1:]).lower()
    for rx, p in PRODUCT_HEAD:
        if re.search(rx, heads):
            return p
    return "general"


def layer_of(relpath):
    """Определяет тематический слой по пути файла."""
    for part in Path(relpath).parts:
        if part in LAYERS:
            return LAYERS[part]
    return "other"


# ---------------------------------------------------------------- главное
def make_chunks(docs):
    """Нарезка: [(путь, текст)] -> (фрагменты, число таблиц, число строк таблиц, проблемы полноты)."""
    TITLES.clear()
    for rel, text in docs:                       # Первый проход: заголовки документов.
        m = re.search(r"^#\s+(.+)$", text, re.M)
        if m:
            TITLES[Path(rel).name] = clean(m.group(1))

    all_chunks, cover_problems = [], []
    tables_total = rows_total = 0
    for rel, text in docs:
        stem = re.sub(r"^\d+-", "", Path(rel).stem)
        layer = layer_of(rel)
        blocks = parse_blocks(text)
        tables_total += sum(1 for b in blocks if b[0] == "t")
        rows_total += sum(len(b[2]) for b in blocks if b[0] == "t")
        doc_title = TITLES.get(Path(rel).name, stem)
        n = 0
        file_chunks = []
        for sec in build_sections(blocks, stem):
            titles = [t for _, t in sec["path"]] or [doc_title]
            sec_type = classify(titles, stem, layer)
            product = product_of(layer, stem, titles)
            for body in pack(sec["units"], stem in PER_ROW_STEMS):
                n += 1
                file_chunks.append({
                    "id": f"{SHORT_ID.get(stem, stem)}-{n:03d}",
                    "file": rel,
                    "layer": layer,
                    "topic": stem,
                    "product": product,
                    "section_type": sec_type,
                    "guard": sec_type in GUARD_TYPES,
                    "path": " › ".join(titles),
                    "text": body,
                    "chars": len(body),
                })
        all_chunks += file_chunks
        # проверка полноты: каждое число исходника должно встретиться в фрагментах файла
        raw_nums = set(re.findall(r"\d+(?:[.,]\d+)?", clean(re.sub(r"(?m)^\s+-{3,}.*$", "", text))))
        got = " ".join(c["path"] + " " + c["text"] for c in file_chunks)
        chunk_nums = set(re.findall(r"\d+(?:[.,]\d+)?", got))
        missing = sorted(raw_nums - chunk_nums)
        if missing:
            cover_problems.append((rel, missing))
    return all_chunks, tables_total, rows_total, cover_problems


def main():
    """Запускает подготовку и сохранение фрагментов из командной строки."""
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "knowledge-base"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "chunks.json"
    docs = load_docs(src)
    all_chunks, tables_total, rows_total, cover_problems = make_chunks(docs)
    report = []

    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    sizes = [c["chars"] for c in all_chunks]
    P = report.append
    P(f"Файлов: {len(docs)}; фрагментов: {len(all_chunks)}; файл: {out}")
    P(f"Таблиц разобрано: {tables_total}; строк таблиц: {rows_total}")
    P(f"Символов в фрагменте: мин {min(sizes)}, медиана {int(statistics.median(sizes))}, "
      f"среднее {int(statistics.mean(sizes))}, макс {max(sizes)} (≈ токенов = символов/3)")
    P(f"Короче 150 символов: {sum(s < 150 for s in sizes)}; длиннее {MAX_CHARS}: {sum(s > MAX_CHARS for s in sizes)}")
    P("\nПо слоям: " + ", ".join(f"{k}={v}" for k, v in Counter(c["layer"] for c in all_chunks).items()))
    P("По продуктам: " + ", ".join(f"{k}={v}" for k, v in Counter(c["product"] for c in all_chunks).most_common()))
    P("По типам разделов: " + ", ".join(f"{k}={v}" for k, v in Counter(c["section_type"] for c in all_chunks).most_common()))
    P(f"Защитных (guard) фрагментов: {sum(c['guard'] for c in all_chunks)}")
    P("\nФрагментов на файл:")
    for rel, cnt in Counter(c["file"] for c in all_chunks).items():
        P(f"  {rel}: {cnt}")
    dup = [t for t, k in Counter(c["text"] for c in all_chunks).items() if k > 1]
    P(f"\nДубликатов текста: {len(dup)}")
    if cover_problems:
        P("ПРОВЕРКА ЧИСЕЛ: не найдены в фрагментах:")
        for rel, miss in cover_problems:
            P(f"  {rel}: {', '.join(miss[:15])}")
    else:
        P("Проверка чисел: все числа из исходных файлов попали во фрагменты.")
    text_report = "\n".join(report)
    (out.parent / "chunks_report.txt").write_text(text_report, encoding="utf-8")
    print(text_report)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
