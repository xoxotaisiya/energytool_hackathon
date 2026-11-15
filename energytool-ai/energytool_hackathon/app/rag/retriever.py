"""Поиск по фрагментам базы знаний: BM25 на чистом Python (без внешних библиотек).

Идея BM25: чем реже слово встречается в базе и чем чаще оно встречается в
конкретном фрагменте, тем сильнее оно «голосует» за этот фрагмент.
Для русского языка используется грубое «стемминг»: слово обрезается до первых
N букв, так «станция/станции/станцию» превращаются в «станц».
"""
import math
import re
from collections import Counter

WORD_RE = re.compile(r"[0-9a-zа-я]+")
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще о из ему теперь когда даже ну ли если уже или ни быть был
него до вас опять уж вам ведь там потом себя ей может они тут где есть надо ней для мы тебя их
чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого
какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда
при наконец два об другой хоть после над больше тот через эти нас про всего них какая много
разве три эту моя свою этой перед иногда лучше чуть том такой им более всегда конечно всю между
это нам наш наши ваш ваши вас нужно можно нельзя""".split())


def tokenize(text, prefix=5):
    """Разбивает текст на нормализованные поисковые слова."""
    out = []
    for w in WORD_RE.findall(text.lower().replace("ё", "е")):
        if w in STOP:
            continue
        if prefix and not w.isdigit() and len(w) > prefix:
            w = w[:prefix]
        out.append(w)
    return out


class BM25Index:
    def __init__(self, chunks, prefix=5, path_weight=2, k1=1.5, b=0.75):
        """Строит BM25-индекс по фрагментам базы знаний."""
        self.chunks, self.prefix, self.k1, self.b = chunks, prefix, k1, b
        self.docs = []
        for c in chunks:
            toks = tokenize(c["path"], prefix) * path_weight + tokenize(c["text"], prefix)
            self.docs.append(Counter(toks))
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=10):
        """Возвращает наиболее релевантные фрагменты для запроса."""
        q = tokenize(query, self.prefix)
        scores = []
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            s = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in order if scores[i] > 0]
