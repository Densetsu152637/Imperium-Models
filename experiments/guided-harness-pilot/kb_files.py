"""Write the knowledge base as plain markdown files (one per topic) plus an index, for the Imperium harness arm."""
import json
import pathlib
import re
import shutil

TOPICS = [
    ("people.md", "People: employers, residences, birth dates", r"works at|employs|lives in|is the home of|born on|date of birth"),
    ("companies.md", "Companies: headquarters, founders, founding years, revenue", r"headquarters|set up by|founder of|reported revenue|was founded in"),
    ("cities-and-countries.md", "Cities and countries: populations, mayors, capitals, currencies", r"is a city in|was established in|mayor|capital of|currency of|population"),
    ("products.md", "Products: makers, prices, weights", r"is made by|sells for|weighs"),
    ("events-distances-rates.md", "Events, road distances, exchange rates, trade units", r"began on|road distance|is worth|Grain Exchange"),
]


def write(dest, kb_path):
    dest = pathlib.Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    kb = [json.loads(line) for line in open(kb_path, encoding="utf-8")]
    buckets = {f: [] for f, _, _ in TOPICS}
    buckets["other-facts.md"] = []
    for p in kb:
        if p.get("topic"):
            buckets.setdefault(p["topic"], []).append(p)
            continue
        for f, _, rx in TOPICS:
            if re.search(rx, p["text"]):
                buckets[f].append(p)
                break
        else:
            buckets["other-facts.md"].append(p)
    index = ["# Knowledge base index", "", "Facts about a fictional world, one fact per line as `[ID] text`.", ""]
    extra = [(f, "Additional facts: " + f[:-3].replace("-", " "), "") for f in buckets
             if f not in {t[0] for t in TOPICS} and f != "other-facts.md"]
    for f, title, _ in TOPICS + [("other-facts.md", "Other facts: calendars, measures, rules", "")] + extra:
        rows = sorted(buckets[f], key=lambda p: p["id"])
        if not rows:
            continue
        (dest / f).write_text(f"# {title}\n\n" + "\n".join(f"[{p['id']}] {p['text']}" for p in rows) + "\n",
                              encoding="utf-8")
        index.append(f"- `{f}`: {title} ({len(rows)} facts)")
    (dest / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    write(sys.argv[1], pathlib.Path(__file__).parent / "suite/kb.jsonl")
