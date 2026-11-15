"""Weighted Reciprocal Rank Fusion: объединяет ранги, а не разные шкалы score."""


def rrf(rankings, weights, k=60):
    if k <= 0 or len(rankings) != len(weights) or any(w < 0 for w in weights) or not any(weights):
        raise ValueError("Некорректные параметры RRF")
    scores = {}
    for hits, weight in zip(rankings, weights):
        seen = set()
        rank = 0
        for index, _ in hits:
            if index in seen:
                continue
            seen.add(index)
            rank += 1
            if weight:
                scores[index] = scores.get(index, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
