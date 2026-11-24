"""Deterministic scoring. No LLM judge."""
import re

ABSTAIN_PHRASES = [
    "unknown", "cannot be determined", "can not be determined", "cannot determine", "can t be determined",
    "can t determine", "not in the knowledge base", "not specified", "not provided", "not stated",
    "not mentioned", "no information", "insufficient information", "not enough information", "unanswerable",
    "abstain", "not available", "cannot be answered", "can t be answered", "cannot answer", "not known",
    "no data", "not given", "not listed",
]
ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}


def strip_markup(s):
    return re.sub(r"[*`_]+", "", s)


def extract_final(text):
    """Return the text after the last 'Final answer:' marker, or None."""
    if not text:
        return None
    clean = strip_markup(text)
    lines = clean.splitlines()
    found = None
    for i, line in enumerate(lines):
        m = re.search(r"final answer\s*[:\-]\s*(.*)$", line, re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            if not ans:
                nxt = [x.strip() for x in lines[i + 1:] if x.strip()]
                ans = nxt[0] if nxt else ""
            found = ans
    return found


def normalize(s):
    s = strip_markup(s or "").lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)           # 1,200 -> 1200
    s = re.sub(r"[^\w\s.]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)         # keep decimal points only
    s = s.replace("_", " ")
    return " ".join(t for t in s.split() if t not in ARTICLES)


def is_abstain(answer):
    n = " " + normalize(answer) + " "
    return any(" " + normalize(p) + " " in n for p in ABSTAIN_PHRASES)


def first_number(answer):
    n = normalize(answer)
    m = re.search(r"-?\d+(?:\.\d+)?", n)
    if m:
        return float(m.group(0))
    for tok in n.split():
        if tok in NUMBER_WORDS:
            return float(NUMBER_WORDS[tok])
    return None


def contains_tokens(answer_norm, gold_norm):
    a, g = answer_norm.split(), gold_norm.split()
    if not g:
        return False
    return any(a[i:i + len(g)] == g for i in range(len(a) - len(g) + 1))


def score(task, text):
    """Return (correct, parsed_answer). `text` is the model output ending with a Final answer line."""
    ans = extract_final(text)
    if ans is None:
        return False, None
    gold, kind = task["answer"], task["answer_type"]
    if kind == "abstain":
        return is_abstain(ans), ans
    if is_abstain(ans):
        return False, ans
    if kind == "number":
        v = first_number(ans)
        return (v is not None and abs(v - float(gold)) <= max(task.get("tol", 0.0), 1e-9) + 1e-6), ans
    golds = [gold] + list(task.get("aliases", []))
    an = normalize(ans)
    return any(contains_tokens(an, normalize(str(g))) for g in golds), ans
