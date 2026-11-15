"""Проверка готового модуля retrieval.py: как работает поиск через retrieve(), в разных вариантах
построения запроса из диалога и в запасном режиме BM25.

Запуск (из папки rag):
    python evaluate_retrieval.py --mode bm25      # без моделей
    python evaluate_retrieval.py --mode dense     # через прокси (нужны LLM_BASE_URL и LLM_API_KEY)
    python evaluate_retrieval.py --mode local     # bge-m3 на своём компьютере (нужен sentence-transformers)
"""
import argparse
import sys

from rag.evaluate import KS, is_gold, load_all
from rag.retrieval import Retriever


def run(retriever, questions):
    """Проверяет качество поиска на выбранном наборе вопросов."""
    hits, rr, misses = {k: 0 for k in KS}, 0.0, []
    for qid, question, gold in questions:
        res = retriever.retrieve(question, top_k=10)
        rank = next((r + 1 for r, c in enumerate(res["chunks"]) if is_gold(c, gold)), None)
        if rank:
            rr += 1 / rank
            for k in KS:
                hits[k] += rank <= k
        if not rank or rank > 5:
            misses.append((qid, rank, res["query"][:80].replace("\n", " "), [c["id"] for c in res["chunks"][:3]]))
    n = len(questions)
    return {k: hits[k] / n for k in KS}, rr / n, misses


def main():
    """Запускает оценку retrieval из командной строки."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Проверка качества поиска через retrieval.py")
    ap.add_argument("--mode", choices=["local", "dense", "bm25"], default="bm25",
                    help="local — bge-m3 на этом компьютере (нужен sentence-transformers); "
                         "dense — bge-m3 через прокси (нужны LLM_BASE_URL, LLM_API_KEY); bm25 — без моделей")
    args = ap.parse_args()
    dense = Retriever(mode=args.mode)
    if dense.mode != args.mode:
        sys.exit(f"Режим {args.mode} не запустился (см. сообщение выше), поэтому проверка остановлена: "
                 "иначе вы увидели бы цифры BM25 под видом эмбеддингов.")
    sets = load_all(dense.chunks)
    if args.mode == "bm25":
        variants = [("BM25, запрос=users", dense, "users", 0.5)]
    else:
        variants = [("dense, запрос=users", dense, "users", 0.5), ("dense, mix, w_last=0.3", dense, "mix", 0.3),
                    ("dense, mix, w_last=0.5", dense, "mix", 0.5), ("dense, mixa, w_last=0.5", dense, "mixa", 0.5),
                    ("dense, mixa, w_last=0.7", dense, "mixa", 0.7),
                    ("BM25, запрос=users", Retriever(mode="bm25"), "users", 0.5)]
    print(f"{'вариант':<24}| " + " | ".join(f"{n:^22}" for n in sets))
    print(f"{'':<24}| " + " | ".join("hit@1 hit@5   MRR     " for _ in sets))
    saved = {}
    for name, r, qm, wl in variants:
        r.query_mode, r.w_last = qm, wl
        row = []
        for sname, qs in sets.items():
            h, mrr, misses = run(r, qs)
            saved[(name, sname)] = misses
            row.append(f"{h[1]:.2f}  {h[5]:.2f}  {mrr:.3f}     ")
        print(f"{name:<24}| " + " | ".join(row))
    main_variant = variants[-1][0] if args.mode == "bm25" else "dense, mixa, w_last=0.5"
    for key in ((main_variant, "dialogs"), (main_variant, "demo")):
        print(f"\nПромахи {key[0]} / {key[1]}: {len(saved[key])}")
        for qid, rank, q, top in saved[key]:
            print(f"- {qid} (позиция эталона: {rank or 'нет в топ-10'}) запрос: {q}… топ-3: {top}")


if __name__ == "__main__":
    main()
