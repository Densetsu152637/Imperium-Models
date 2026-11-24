"""Arms A (plain) and B (rag), plus the shared prompts."""
import llm

SYSTEM = ("You are a careful assistant. Answer the question. You may reason briefly, step by step. Then end with a "
          "final line exactly of the form\nFinal answer: <answer>\nwhere <answer> is as short as possible: a number "
          "(no units, no thousands separators), a name, or a short phrase. If you do not know the answer, or it "
          "cannot be determined from the information you have, end with\nFinal answer: Unknown")

RAG_NOTE = ("Reference passages follow. They describe a fictional world; where they conflict with what you remember "
            "from the real world, the passages take priority. If the question concerns this fictional world and the "
            "passages do not contain what is needed, answer Unknown. Questions about the real world or pure "
            "arithmetic or logic can be answered from your own knowledge.")


def rag_messages(question, hits):
    block = "\n".join(f"[{d}] {t}" for _, d, t in hits)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{RAG_NOTE}\n\nPassages:\n{block}\n\nQuestion: {question}"}]


def plain_messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question: {question}"}]


def run_plain(task, bm25, meter):
    text = llm.chat(plain_messages(task["question"]), meter, temperature=0.0)
    return {"output": text, "final_text": text}


def run_rag(task, bm25, meter, k=5):
    hits = bm25.search(task["question"], k)
    text = llm.chat(rag_messages(task["question"], hits), meter, temperature=0.0)
    return {"output": text, "final_text": text, "retrieved": [d for _, d, _ in hits]}
