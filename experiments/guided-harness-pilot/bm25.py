"""Minimal BM25 over kb.jsonl (stdlib only)."""
import json
import math
import re
from collections import Counter

STOP = set("a an the of in on at to for is are was were be by and or as with that this which who whom what where "
           "when how many much does do did has have had it its from than then if about into per each".split())


def tokens(text):
    out = []
    for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())):
        if t not in STOP:
            out.append(t)
    return out


class BM25:
    def __init__(self, kb_path, k1=1.5, b=0.75):
        self.docs = [json.loads(line) for line in open(kb_path, encoding="utf-8")]
        self.text = {d["id"]: d["text"] for d in self.docs}
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d["text"])) for d in self.docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, k=5):
        q = set(tokens(query))
        scored = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = c.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, self.docs[i]["id"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, d, self.text[d]) for s, d in scored[:k]]
