"""Scorer self-test. Run: python test_scorer.py"""
from scoring import extract_final, score

S = {"answer": "Ishkar Glass", "answer_type": "string"}
N = {"answer": 36, "answer_type": "number", "tol": 0.0}
F = {"answer": 66.67, "answer_type": "number", "tol": 0.01}
U = {"answer": "unknown", "answer_type": "abstain"}
cases = [
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
