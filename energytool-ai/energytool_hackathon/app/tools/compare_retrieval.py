"""Сравнение режимов: fallback не выдаётся за результат embeddings/реранкера."""
import argparse
import json
import os
import time
from pathlib import Path
from rag.retrieval import Retriever
from rag.evaluate import load_all, is_gold
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", nargs="+", default=["bm25", "ollama", "hybrid-ollama"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--sets", nargs="+", choices=["demo", "own", "dialogs"], default=["demo", "own", "dialogs"])
    ap.add_argument("--output", type=Path, default=config.ROOT / "reports/retrieval_comparison.json")
    args = ap.parse_args()
    if args.rerank:
        os.environ["RERANKER_ENABLED"] = "1"
    report = {"variants": [], "metric": "Hit@k: хотя бы один эталонный раздел; не качество ответов LLM",
              "reranker_backend": os.getenv("RERANKER_BACKEND", "api") if args.rerank else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        r = Retriever(kb_path=config.KB_PATH or None, mode=mode)
        row = {"requested_mode": mode, "actual_mode": r.mode, "startup_s": round(r.startup_s, 2),
               "reranker_requested": args.rerank, "completed": False, "candidate_k": r.candidate_k,
               "embedding_model": getattr(r.embedder, "model_name", None),
               "reranker_model": getattr(r.reranker, "model", None)}
        row["queries"] = []
        report["variants"].append(row)
        try:
            if r.mode != mode or args.rerank and r.reranker is None:
                row["skipped"] = "Поиск или реранкер недоступен"
            else:
                sets = {}
                questions_by_set = {name: questions for name, questions in load_all(r.chunks).items() if name in args.sets}
                warmup = time.perf_counter()
                r.prefetch([q for questions in questions_by_set.values() for _, q, _ in questions])
                row["query_warmup_s"] = round(time.perf_counter() - warmup, 2)
                for name, questions in questions_by_set.items():
                    ranks, times = [], []
                    for qid, q, gold in questions:
                        result = r.retrieve(q, top_k=10)
                        rank = next((i + 1 for i, c in enumerate(result["chunks"]) if is_gold(c, gold)), None)
                        row["queries"].append({"set": name, "id": qid, "rank": rank,
                                               "ids": [c["id"] for c in result["chunks"]],
                                               "mode": result["mode"], "reranked": result["reranked"],
                                               "rerank_error": result.get("rerank_error"),
                                               "search_ms": result["search_ms"]})
                        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"{mode}/{name}/{qid}: rank={rank}, reranked={result['reranked']}", flush=True)
                        if result["mode"] != mode or args.rerank and not result["reranked"]:
                            raise RuntimeError("Fallback во время сравнения")
                        ranks.append(rank)
                        times.append(result["search_ms"])
                    n = len(ranks)
                    sets[name] = {"n": n, "hit1": sum(x == 1 for x in ranks) / n,
                                  "hit5": sum(x is not None and x <= 5 for x in ranks) / n,
                                  "hit10": sum(x is not None and x <= 10 for x in ranks) / n,
                                  "mrr": sum(1 / x if x else 0 for x in ranks) / n,
                                  "mean_search_ms": sum(times) / n}
                row.update({"completed": True, "sets": sets})
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        finally:
            r.close()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if all(row["completed"] for row in report["variants"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
