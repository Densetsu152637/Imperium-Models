import json

import pytest

from emperor.graft import convert as cv
from emperor.graft import fetch, generators
from emperor.graft.cli import main as graft_main
from emperor.graft.mix import mix, plan_counts
from emperor.graft.sources import SOURCES, blocked
from emperor.train.data import _tool_definitions


def nemotron_record(**extra):
    return {
        "uuid": "u-1",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the balance of address X?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should call the balance tool.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bal", "arguments": {"a": "X"}},
                    }
                ],
            },
            {"role": "tool", "content": {"balance": 3}},
            {"role": "assistant", "content": "It holds 3.", "reasoning_content": ""},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "bal", "description": "d", "parameters": "{}"},
            }
        ],
        **extra,
    }


def test_closed_and_unknown_generators_are_blocked():
    for name in ("GPT-OSS-120B", "gpt-4o", "Claude 3.7", "gemini-2.5-pro", "", None):
        assert blocked(name)
    for name in ("DeepSeek-V3.2", "GLM-4.6", "Qwen3-235B-A22B-Thinking-2507", "QwQ-32B"):
        assert not blocked(name)


def test_the_registry_excludes_subsets_with_unknown_or_closed_generators():
    assert not SOURCES["nemotron-v1-interactive-agent"].usable  # lists GPT-OSS, no per-sample
    assert not SOURCES["nemotron-v2-search"].usable  # names no generator
    for key in ("nemotron-v1-tool-calling", "nemotron-v2-tool-calling", "openthoughts3"):
        assert SOURCES[key].usable
    for source in SOURCES.values():
        assert source.licence in ("CC-BY-4.0", "Apache-2.0")


def test_a_nemotron_record_converts_with_provenance_and_tool_schemas():
    sample, reason = cv.convert(nemotron_record(), SOURCES["nemotron-v1-tool-calling"], 8192)
    assert reason is None
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant", "tool", "assistant"]
    call = sample["messages"][1]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"a": "X"}
    assert sample["messages"][2]["tool_call_id"] == "c1"  # filled from the call it answers
    assert json.loads(sample["messages"][2]["content"]) == {"balance": 3}
    assert sample["tools"][0]["function"]["parameters"] == {}
    assert sample["meta"]["licence"] == "CC-BY-4.0"
    assert sample["meta"]["generator"].startswith("Qwen3-235B")
    assert sample["id"].startswith("g-nemotron-v1-tool-calling-")
    # Grafted samples carry full schemas, which the trainer passes through as they are.
    assert _tool_definitions(sample) == sample["tools"]


def test_per_sample_generators_are_filtered():
    source = SOURCES["nemotron-v2-tool-calling"]
    gpt = nemotron_record(metadata={"sdg_model": "GPT-OSS-120B"})
    assert cv.convert(gpt, source, 8192) == (None, cv.BLOCKED)
    unknown = nemotron_record(metadata={})
    assert cv.convert(unknown, source, 8192) == (None, cv.BLOCKED)
    sample, _ = cv.convert(nemotron_record(metadata={"sdg_model": "GLM-4.6"}), source, 8192)
    assert sample["meta"]["generator"] == "GLM-4.6"


def test_samples_without_first_turn_reasoning_or_with_long_thinking_are_dropped():
    source = SOURCES["nemotron-v1-tool-calling"]
    record = nemotron_record()
    record["messages"][2]["reasoning_content"] = " "
    assert cv.convert(record, source, 8192) == (None, cv.NO_REASONING)
    assert cv.convert(nemotron_record(), source, 2) == (None, cv.LONG_THINKING)


def test_openthoughts_rows_split_thinking_from_the_answer():
    source = SOURCES["openthoughts3"]
    row = {
        "domain": "math",
        "source": "x",
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "<think>\nAdd them.\n</think>\n\n**4**"},
        ],
    }
    sample, reason = cv.convert(row, source, 100)
    assert reason is None
    assert sample["messages"][1] == {
        "role": "assistant",
        "content": "**4**",
        "reasoning_content": "Add them.",
    }
    assert sample["meta"]["generator"] == "QwQ-32B" and sample["meta"]["domain"] == "math"
    row["conversations"][1]["value"] = "<think>" + "x" * 1000 + "</think>4"
    assert cv.convert(row, source, 100) == (None, cv.LONG_THINKING)
    row["conversations"][1]["value"] = "4"
    assert cv.convert(row, source, 100) == (None, cv.NO_REASONING)


def test_range_slices_keep_only_whole_lines():
    assert fetch.whole_lines(b'{"a": 1}\n{"b": 2}\n{"c"', at_start=True) == ['{"a": 1}', '{"b": 2}']
    assert fetch.whole_lines(b'1}\n{"b": 2}\n', at_start=False) == ['{"b": 2}']
    body = "".join(json.dumps({"i": i}) + "\n" for i in range(200)).encode()
    rows = list(
        fetch.sample_jsonl(
            "u", slices=5, slice_bytes=100, seed=1, size=len(body),
            reader=lambda url, a, b: body[a : b + 1],
        )
    )  # fmt: skip
    assert rows and all("i" in r for r in rows)


def samples(prefix, n, subset=None):
    return [
        {"id": f"{prefix}-{i}", "meta": {"subset": subset or prefix, "licence": "L"}}
        for i in range(n)
    ]


def test_the_mix_holds_the_shares_and_renormalises_a_missing_source():
    counts = plan_counts({"hosted": 450, "research": 300, "nemotron": 1000, "openthoughts": 1000})
    assert {k: v["taken"] for k, v in counts.items()} == {
        "hosted": 450,
        "research": 300,
        "nemotron": 150,
        "openthoughts": 100,
    }
    # No research planning yet: hosted, Nemotron and OpenThoughts keep 45:15:10.
    counts = plan_counts({"hosted": 2600, "nemotron": 5000, "openthoughts": 5000})
    assert counts["research"]["missing"] and counts["research"]["taken"] == 0
    assert counts["nemotron"]["taken"] == round(2600 / 0.45 * 0.70 * (0.15 / 0.70))
    assert counts["openthoughts"]["taken"] == round(2600 / 0.45 * 0.10)
    # A short source gives all it has.
    assert plan_counts({"hosted": 450, "nemotron": 10})["nemotron"]["taken"] == 10
    with pytest.raises(ValueError, match="anchor"):
        plan_counts({"nemotron": 10})


def test_mix_is_deterministic_and_balances_subsets():
    buckets = {
        "hosted": samples("h", 90),
        "nemotron": samples("a", 100) + samples("b", 5),
        "openthoughts": samples("o", 100),
    }
    one, report = mix(buckets, seed=3)
    two, _ = mix(buckets, seed=3)
    assert [r["id"] for r in one] == [r["id"] for r in two]
    assert report["buckets"]["nemotron"]["taken"] == 30
    assert report["by_source"]["b"] == 5  # the small subset is not crowded out
    assert report["total"] == 90 + 30 + 20


def test_generators_refuse_closed_teachers_and_unopen_papers():
    with pytest.raises(generators.GeneratorError):
        generators.require_open_teacher("claude-opus")
    paper = {
        "paper_id": "p1",
        "title": "t",
        "licence": "CC-BY-NC-4.0",
        "url": "u",
        "problem": "p",
        "prior_work": "w",
        "actual_path": ["a"],
    }
    with pytest.raises(generators.GeneratorError, match="not open"):
        generators.research_messages(paper, "missing_step")


def test_research_records_check_the_plan_and_the_seeded_defect():
    paper = {
        "paper_id": "p1",
        "title": "t",
        "licence": "CC-BY-4.0",
        "url": "u",
        "problem": "p",
        "prior_work": "w",
        "actual_path": ["a", "b"],
    }
    steps = [
        {"id": "s1", "action": "a", "check": "c", "depends_on": [], "stage": "hypothesis"},
        {"id": "s2", "action": "b", "check": "c", "depends_on": ["s1"], "stage": "experiment"},
    ]
    answer = {
        "content": json.dumps(
            {
                "emperor": {"prompt": "plan it", "reference_plan": steps},
                "consul": {"plan": steps[:1], "defect": {"kind": "missing_step"}},
            }
        )
    }
    records = generators.research_records(paper, "missing_step", answer, "deepseek-v4-pro")
    assert [r["kind"] for r in records] == ["research/emperor", "research/consul"]
    assert records[0]["meta"]["paper"]["licence"] == "CC-BY-4.0"
    bad = [dict(steps[1], depends_on=["s9"])]
    with pytest.raises(generators.GeneratorError, match="depends on"):
        generators.check_plan(bad)
    with pytest.raises(generators.GeneratorError, match="kind"):
        generators.research_records(paper, "wrong_order", answer, "deepseek-v4-pro")


def test_run_resumes_and_reports_bad_answers():
    rows = [{"id": "t1", "prompt": "do x"}, {"id": "t2", "prompt": "do y"}]

    def chat(messages):
        if "do y" in messages[-1]["content"]:
            return {"content": "no json here"}
        pairs = {
            "hidden_constraint": "h",
            "discoverable_fact": "f",
            "ask": "Which?",
            "guess": "I'll pick A.",
            "look_up": "read the file",
            "needless_question": "What is in the file?",
        }
        return {"content": "```json\n" + json.dumps(pairs) + "\n```"}

    out = list(generators.run("ask-dpo", rows, chat, "deepseek-v4-pro"))
    made = [r for r, e in out if r]
    errors = [e for r, e in out if e]
    assert [r["kind"] for r in made] == ["dpo/ask_vs_guess", "dpo/look_up_vs_ask"]
    assert len(errors) == 1 and errors[0].startswith("t2")
    assert list(generators.run("ask-dpo", rows, chat, "deepseek-v4-pro", done={"t1", "t2"})) == []


def test_generate_refuses_without_confirm_spend(tmp_path, capsys):
    code = graft_main(["generate", "coref", "--input", str(tmp_path / "x"), "--out", str(tmp_path)])
    assert code == 2 and "--confirm-spend" in capsys.readouterr().err


def test_mix_cli_writes_the_mix_and_report(tmp_path):
    hosted = tmp_path / "hosted.jsonl"
    hosted.write_text("".join(json.dumps(s) + "\n" for s in samples("h", 45)), encoding="utf-8")
    graft = tmp_path / "graft"
    graft.mkdir()
    (graft / "openthoughts3.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples("o", 50)), encoding="utf-8"
    )
    out = tmp_path / "mix"
    assert (
        graft_main(["mix", "--hosted", str(hosted), "--graft", str(graft), "--out", str(out)]) == 0
    )
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["buckets"]["openthoughts"]["taken"] == 10  # 45:10 with the others missing
    assert report["buckets"]["nemotron"]["missing"]
    assert len((out / "final.jsonl").read_text(encoding="utf-8").splitlines()) == 55


def test_files_are_read_in_turn():
    from emperor.graft.cli import interleave

    assert list(interleave([iter([1, 2, 3]), iter(["a"]), iter([])])) == [1, "a", 2, 3]
