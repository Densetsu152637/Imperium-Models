"""Deterministic strict scoring. No LLM judge.

Rules (all applied to the text after the last `Final answer:` marker on its line):
 1. A missing marker is wrong.
 2. Abstain items: correct only if the line abstains (a phrase such as "Unknown" or "not in the knowledge base")
    and gives no concrete answer (no digit, no hedge word such as "likely" or "but").
 3. A non-abstain item whose line abstains is wrong.
 4. Number items: after removing numbers listed in the item's `allow_numbers`, the line must contain exactly one
    number (or, when the line has "=", the number after the last "="); ranges and several candidates fail. The
    number must be within the item's tolerance of the gold.
 5. Text items: the gold (or one of the item's `aliases`) must appear as a whole-token run; the line fails if it
    also offers an alternative: "or", "either", "/", "and/or", or a list of short capitalised candidates; or if any
    of the item's `distractors` appears in it.
"""
import re

ABSTAIN_PHRASES = [
    "unknown", "cannot be determined", "can not be determined", "cannot determine", "can t be determined",
    "can t determine", "not in the knowledge base", "not specified", "not provided", "not stated",
    "not mentioned", "no information", "insufficient information", "not enough information", "unanswerable",
    "abstain", "not available", "cannot be answered", "can t be answered", "cannot answer", "not known",
    "no data", "not given", "not listed",
]
HEDGE_WORDS = {"but", "however", "likely", "probably", "possibly", "perhaps", "maybe", "approximately", "around",
               "guess", "assuming", "estimated", "roughly", "about"}
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


def numbers_in(answer):
    """All numbers in the answer, digits first, then number words when there are no digits."""
    n = normalize(answer)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)]
    if not nums:
        nums = [float(NUMBER_WORDS[t]) for t in n.split() if t in NUMBER_WORDS]
    return nums


def first_number(answer):
    nums = numbers_in(answer)
    return nums[0] if nums else None


def contains_tokens(answer_norm, gold_norm):
    a, g = answer_norm.split(), gold_norm.split()
    if not g:
        return False
    return any(a[i:i + len(g)] == g for i in range(len(a) - len(g) + 1))


def has_alternative(ans, gold_norms):
    """True when the line offers more than one candidate answer."""
    low = " " + strip_markup(ans).lower() + " "
    if re.search(r"\s(or|either|and/or)\s|/", low):
        return True
    parts = [p.strip() for p in re.split(r",|;|\sand\s|&", strip_markup(ans)) if p.strip()]
    if len(parts) >= 2:
        hit = [p for p in parts if any(contains_tokens(normalize(p), g) for g in gold_norms)]
        others = [p for p in parts if p not in hit]
        if hit and any(len(p.split()) <= 4 and p[:1].isupper() for p in others):
            return True
    return False


def score(task, text):
    """Return (correct, parsed_answer). `text` is the model output ending with a Final answer line."""
    ans = extract_final(text)
    if ans is None:
        return False, None
    gold, kind = task["answer"], task["answer_type"]
    ab = is_abstain(ans)
    if kind == "abstain":
        concrete = bool(re.search(r"\d", ans)) or bool(set(normalize(ans).split()) & HEDGE_WORDS)
        return (ab and not concrete), ans
    if ab:
        return False, ans
    if kind == "number":
        nums = numbers_in(ans)
        allowed = {float(x) for x in task.get("allow_numbers", [])}
        if "=" in ans and re.search(r"\d", ans.rsplit("=", 1)[1]):
            nums = numbers_in(ans.rsplit("=", 1)[1])
        nums = [x for x in nums if x not in allowed]
        if len(nums) != 1:
            return False, ans
        return abs(nums[0] - float(gold)) <= max(task.get("tol", 0.0), 1e-9) + 1e-6, ans
    golds = [normalize(str(g)) for g in [gold] + list(task.get("aliases", []))]
    an = normalize(ans)
    if not any(contains_tokens(an, g) for g in golds):
        return False, ans
    if has_alternative(ans, golds):
        return False, ans
    for d in task.get("distractors", []):
        dn = normalize(str(d))
        if dn and contains_tokens(an, dn) and not any(contains_tokens(g, dn) for g in golds):
            return False, ans
    return True, ans
