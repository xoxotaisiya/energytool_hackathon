"""Как ведёт себя «уверенность» модуля retrieval: у вопросов с ответом в базе и без него.

Нужно, чтобы выбрать порог отказа. Запуск (из папки rag):
    python measure_scores.py --mode dense      # через прокси (LLM_BASE_URL, LLM_API_KEY)
    python measure_scores.py --mode local      # bge-m3 на своём компьютере (sentence-transformers)
"""
import argparse
import json
import sys

from evaluate import HERE, is_gold, load_all
from rag.retrieval import Retriever

OUT_OF_BASE = {"q30", "q31"}      # наши вопросы, на которые в базе ответа нет


def main():
    """Считает и выводит метрики качества поиска."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Уверенность поиска и порог отказа")
    ap.add_argument("--mode", choices=["local", "dense"], default="local",
                    help="local — bge-m3 на этом компьютере; dense — через прокси (LLM_BASE_URL, LLM_API_KEY)")
    args = ap.parse_args()
    r = Retriever(mode=args.mode)
    if r.mode != args.mode:
        sys.exit(f"Режим {args.mode} не запустился (см. сообщение выше): без эмбеддингов уверенности нет.")
    rows = {"есть ответ": [], "нет ответа (q30,q31)": [], "вне темы (o01-o14)": []}
    for sname, qs in load_all(r.chunks).items():
        for qid, q, gold in qs:
            res = r.retrieve(q)
            kind = "нет ответа (q30,q31)" if qid in OUT_OF_BASE else "есть ответ"
            rows[kind].append((res["confidence"], qid, q, is_gold(res["chunks"][0], gold), res["chunks"][0]["id"]))
    for it in json.loads((HERE / "eval_out.json").read_text(encoding="utf-8")):
        res = r.retrieve(it["question"])
        rows["вне темы (o01-o14)"].append((res["confidence"], it["id"], it["question"], False, res["chunks"][0]["id"]))
    for kind, lst in rows.items():
        lst.sort()
        vals = [x[0] for x in lst]
        print(f"\n=== {kind}: {len(lst)}; уверенность: мин {min(vals):.3f}, медиана {vals[len(vals)//2]:.3f}, макс {max(vals):.3f}")
        for conf, qid, q, ok, cid in (lst if len(lst) <= 20 else lst[:8]):
            print(f"  {conf:.3f} {qid:<5} {' '.join(q.split())[-58:]:<58} {cid}")
    thr = [0.45, 0.50, 0.55]
    ans, off = [x[0] for x in rows["есть ответ"]], [x[0] for x in rows["вне темы (o01-o14)"]]
    print("\nпорог | вопросов с ответом ошибочно отказано | вне темы отсечено")
    for t in thr:
        print(f"{t:.2f}  | {sum(v < t for v in ans)} из {len(ans)} | {sum(v < t for v in off)} из {len(off)}")


if __name__ == "__main__":
    main()
