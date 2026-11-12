"""Оценка качества поиска на проверочных вопросах.

Запуск: python evaluate.py
Метрики: hit@k — доля вопросов, у которых хотя бы один эталонный фрагмент попал в топ-k;
MRR — средняя величина 1/позиция первого эталонного фрагмента (0, если не найден).
"""
import csv
import json
import sys
from pathlib import Path

import os

from rag.build_chunks import load_docs, make_chunks
from rag.retriever import BM25Index

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5, 10)


def load_chunks():
    """Фрагменты базы: из KB_PATH (папка или zip) либо из ../knowledge-base рядом с папкой rag."""
    return make_chunks(load_docs(os.getenv("KB_PATH") or HERE.parent / "knowledge-base"))[0]


SETS = {"demo": "eval_demo.json",   # 18 вопросов из выданной организаторами выборки
        "own": "eval_own.json",     # 31 вопрос, придуманный нами
        "dialogs": "eval_dialogs.json"}   # 8 диалогов с уточнением в конце, придуманы нами


def load_all(chunks):
    """{'demo': [...], 'own': [...]} — два набора, которые считаются отдельно."""
    return {name: load_questions(chunks, file) for name, file in SETS.items()}


def load_questions(chunks, file="eval_demo.json"):
    """Загружает вопросы и эталонные фрагменты для оценки."""
    demo = {}
    with (HERE.parent / "demo_questions.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            demo[int(row["id"])] = row["question"]
    items = json.loads((HERE / file).read_text(encoding="utf-8"))
    out = []
    for it in items:
        qid = f"demo{it['demo_id']}" if "demo_id" in it else it["id"]
        question = demo[it["demo_id"]] if "demo_id" in it else it["question"]
        gold = [tuple(g.split("|", 1)) for g in it["gold"]]
        for topic, sub in gold:       # защита от опечаток в эталоне
            if not any(c["topic"] == topic and sub in c["path"] for c in chunks):
                print(f"ВНИМАНИЕ: эталон {topic}|{sub} в {qid} не соответствует ни одному фрагменту")
        out.append((qid, question, gold))
    return out


def is_gold(chunk, gold):
    """Проверяет, соответствует ли найденный фрагмент эталону."""
    return any(chunk["topic"] == t and s in chunk["path"] for t, s in gold)


def evaluate(index, questions, query_fn=lambda q: q):
    """Считает hit@k, MRR и список поисковых промахов."""
    hits = {k: 0 for k in KS}
    rr, misses = 0.0, []
    for qid, question, gold in questions:
        res = index.search(query_fn(question), 10)
        rank = next((r + 1 for r, (i, _) in enumerate(res) if is_gold(index.chunks[i], gold)), None)
        if rank:
            rr += 1 / rank
            for k in KS:
                hits[k] += rank <= k
        if not rank or rank > 5:
            misses.append((qid, question, rank, [index.chunks[i]["id"] for i, _ in res[:5]]))
    n = len(questions)
    return {k: hits[k] / n for k in KS}, rr / n, misses


def main():
    """Запускает сравнение настроек BM25 на тестовых наборах."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    chunks = load_chunks()
    sets = load_all(chunks)
    print("Вопросов: " + ", ".join(f"{n} — {len(q)}" for n, q in sets.items()) + f"; фрагментов: {len(chunks)}\n")
    print("Параметры подбираются на «own», а «demo» — независимая проверка.\n")
    print("prefix path_w | " + " | ".join(f"{n:^28}" for n in sets))
    print("              | " + " | ".join("hit@1 hit@5 hit@10   MRR " for _ in sets))
    best = None
    for prefix in (0, 4, 5, 6, 7):
        for pw in (1, 2, 3):
            idx = BM25Index(chunks, prefix, pw)
            res = {n: evaluate(idx, q) for n, q in sets.items()}
            row = " | ".join(f"{h[1]:.2f}  {h[5]:.2f}  {h[10]:.2f}   {mrr:.3f}" for h, mrr, _ in res.values())
            print(f"{prefix:>6} {pw:>6} | {row}")
            h, mrr, _ = res["own"]
            if best is None or (h[5], mrr) > best[0]:
                best = ((h[5], mrr), prefix, pw)
    _, prefix, pw = best
    print(f"\nЛучшая конфигурация по «own»: prefix={prefix}, path_weight={pw}")
    by_id = {c["id"]: c for c in chunks}
    for name, questions in sets.items():
        _, _, misses = evaluate(BM25Index(chunks, prefix, pw), questions)
        print(f"\n[{name}] Промахи (нет эталона в топ-5): {len(misses)}")
        for qid, q, rank, top in misses:
            short = " ".join(q.split())[:90]
            print(f"- {qid} (позиция эталона: {rank or 'нет в топ-10'}) {short}")
            print("    топ-5: " + ", ".join(f"{t} [{by_id[t]['path'].split(' › ')[-1][:28]}]" for t in top))


if __name__ == "__main__":
    main()
