"""Scorer self-test. Run: python test_scorer.py

Old cases are unchanged except where the stricter rules intentionally flip them (marked NEW-RULE).
"""
from scoring import extract_final, score

S = {"answer": "Ishkar Glass", "answer_type": "string", "aliases": ["Ishkar"], "distractors": ["Tessel Optics"]}
N = {"answer": 36, "answer_type": "number", "tol": 0.0}
F = {"answer": 66.67, "answer_type": "number", "tol": 0.01}
U = {"answer": "unknown", "answer_type": "abstain"}
W = {"answer": "William Shakespeare", "answer_type": "string", "aliases": ["Shakespeare"]}
cases = [
    # old cases
    (S, "Reasoning...\nFinal answer: Ishkar Glass", True),
    (S, "**Final answer:** The Ishkar Glass company.", True),
    (S, "Final answer: ishkar glass.", True),
    (S, "Final answer: Tessel Optics", False),
    (S, "Final answer: Unknown", False),
    (S, "no final line here", False),
    (N, "Final answer: 36 days", True),
    (N, "Final answer: 1,036", False),
    (N, "Final answer: thirty", False),
    (F, "Final answer: $66.66", True),
    (F, "Final answer: 66.5", False),
    (U, "Final answer: Unknown", True),
    (U, "Final answer: The date of birth is not in the knowledge base.", True),
    (U, "Final answer: It cannot be determined", True),
    (U, "Final answer: 12 March 1970", False),
    ({"answer": "no", "answer_type": "string"}, "Final answer: No", True),
    ({"answer": "Neil Armstrong", "answer_type": "string"}, "Final answer: Neil Young", False),
    ({"answer": 2, "answer_type": "number", "tol": 0.001}, "Final answer: two", True),
    # new rules
    (S, "Final answer: Ishkar Glass or Tessel Optics", False),                    # hedge with "or" and a distractor
    (S, "Final answer: either Ishkar Glass or Ishkar Foods", False),               # hedge, no listed distractor
    (S, "Final answer: Ishkar Glass / Tessel Optics", False),                      # slash alternative
    (S, "Final answer: Ishkar Glass, Tessel Optics", False),                       # list of candidates
    (S, "Final answer: Ishkar Glass (not Tessel Optics)", False),                  # distractor present
    (S, "Final answer: Ishkar", True),                                             # alias
    ({"answer": "no", "answer_type": "string", "distractors": ["yes"]}, "Final answer: No, the bishop cannot", True),
    ({"answer": "no", "answer_type": "string", "distractors": ["yes"]}, "Final answer: Yes", False),
    (N, "Final answer: between 36 and 40", False),                                 # range
    (N, "Final answer: 36-40", False),                                             # range with dash
    (N, "Final answer: 36 or 40", False),                                          # several candidates
    (N, "Final answer: 6 * 6 = 36", True),                                         # number after "="
    (N, "Final answer: 36 days (about 5 weeks)", False),                           # two numbers
    ({"answer": 36, "answer_type": "number", "tol": 0.0, "allow_numbers": [2024]}, "Final answer: 36 in 2024", True),
    (W, "Final answer: Shakespeare", True),                                        # NEW-RULE: alias, was wrong before
    (W, "Final answer: Shakespeare or Marlowe", False),
    (U, "Final answer: Unknown, but probably 1970", False),                        # abstain plus a concrete answer
    (U, "Final answer: Unknown (maybe Tessel Optics)", False),
    (S, "Final answer: Unknown, possibly Ishkar Glass", False),                    # abstain on a non-abstain item
]
bad = 0
for task, text, want in cases:
    got, _ = score(task, text)
    if got != want:
        bad += 1
        print("FAIL", task["answer"], repr(text), "want", want, "got", got)
assert extract_final("a\nFinal answer: x\nFinal answer: y") == "y"
print(f"{len(cases) - bad}/{len(cases)} scorer cases passed")
raise SystemExit(1 if bad else 0)
