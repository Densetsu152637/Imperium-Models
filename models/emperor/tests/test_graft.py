import json

import pytest

from emperor.data.teacher import TeacherOutputError
from emperor.graft import convert as cv
from emperor.graft import fetch, generators
from emperor.graft.cli import DEFAULT_LEDGER, interleave
from emperor.graft.cli import main as graft_main
from emperor.graft.mix import mix, plan_counts
from emperor.graft.sources import SOURCES, allowed
from emperor.release import card
from emperor.train.config import parse_config
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


def ot_row(source="nvidia/OpenCodeReasoning", answer="<think>\nAdd them.\n</think>\n\n**4**"):
    return {
        "domain": "code",
        "source": source,
        "conversations": [
            {"from": "human", "value": "Write a function that adds two numbers."},
            {"from": "gpt", "value": answer},
        ],
    }


def test_generators_are_an_allow_list():
    for name in (
        "DeepSeek-V3.2",
        "deepseek/DeepSeek-V3.2",
        "DeepSeek-v3.2",
        "GLM-4.6",
        "QwQ-32B",
        "Qwen3-235B-A22B-Thinking-2507+Qwen3-235B-A22B-Instruct-2507",
    ):
        assert allowed(name), name
    for name in ("GPT-OSS-120B", "gpt-4o", "Claude 3.7", "gemini-2.5-pro", "Kimi-K2", "", None):
        assert not allowed(name), name
    assert not allowed("Qwen3-32B+GPT-OSS-120B")  # every named model must be allowed


def test_the_registry_excludes_unattributable_subsets_and_pins_revisions():
    assert not SOURCES["nemotron-v1-interactive-agent"].usable  # lists GPT-OSS, no per-sample
    assert not SOURCES["nemotron-v2-search"].usable  # names no generator
    assert not SOURCES["nemotron-v2-customer-service"].usable  # user/tool turns unattributed
    for key in ("nemotron-v1-tool-calling", "nemotron-v2-tool-calling", "openthoughts3"):
        assert SOURCES[key].usable
    for source in SOURCES.values():
        assert source.licence in ("CC-BY-4.0", "Apache-2.0")
        assert len(source.revision) == 40 and source.revision in source.url
    assert fetch.file_url("a/b", "f.jsonl", "abc") == (
        "https://huggingface.co/datasets/a/b/resolve/abc/f.jsonl"
    )


def test_a_nemotron_record_converts_with_provenance_and_tool_schemas():
    record = nemotron_record(metadata={"sdg_model": "GLM-4.6", "source": "tool_eyes"})
    sample, reason = cv.convert(record, SOURCES["nemotron-v2-tool-calling"], 8192)
    assert reason is None
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant", "tool", "assistant"]
    call = sample["messages"][1]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"a": "X"}
    assert sample["messages"][2]["tool_call_id"] == "c1"  # filled from the call it answers
    assert json.loads(sample["messages"][2]["content"]) == {"balance": 3}
    assert sample["tools"][0]["function"]["parameters"] == {}
    meta = sample["meta"]
    assert meta["licence"] == "CC-BY-4.0" and meta["generator"] == "GLM-4.6"
    assert meta["upstream_source"] == "tool_eyes" and meta["judged"] is False
    assert meta["revision"] == SOURCES["nemotron-v2-tool-calling"].revision
    # Grafted samples carry full schemas, which the trainer passes through as they are.
    assert _tool_definitions(sample) == sample["tools"]


def test_generators_not_on_the_allow_list_are_dropped():
    source = SOURCES["nemotron-v2-tool-calling"]
    for model in ("GPT-OSS-120B", "Kimi-K2", None):
        record = nemotron_record(metadata={"sdg_model": model} if model else {})
        assert cv.convert(record, source, 8192) == (None, cv.BLOCKED)


def test_schema_checks_catch_bad_calls_and_secrets():
    source = SOURCES["nemotron-v1-tool-calling"]
    record = nemotron_record()
    record["messages"][2]["tool_calls"][0]["function"]["name"] = "unknown_tool"
    assert cv.convert(record, source, 8192) == (None, "schema:tool_not_offered")
    record = nemotron_record()
    record["messages"][4]["content"] = "Your key is sk-ant-api03-" + "a" * 40
    _, reason = cv.convert(record, source, 8192)
    assert reason == "schema:secret_in_assistant"
    record = nemotron_record()
    del record["messages"][3]  # the call is never answered
    _, reason = cv.convert(record, source, 8192)
    assert reason == "schema:unanswered_tool_call"


def test_samples_without_first_turn_reasoning_or_with_long_thinking_are_dropped():
    source = SOURCES["nemotron-v1-tool-calling"]
    record = nemotron_record()
    record["messages"][2]["reasoning_content"] = " "
    assert cv.convert(record, source, 8192) == (None, cv.NO_REASONING)
    assert cv.convert(nemotron_record(), source, 2) == (None, cv.LONG_THINKING)


def test_openthoughts_keeps_only_allowed_upstream_sources():
    source = SOURCES["openthoughts3"]
    sample, reason = cv.convert(ot_row(), source, 100)
    assert reason is None
    assert sample["messages"][1] == {
        "role": "assistant",
        "content": "**4**",
        "reasoning_content": "Add them.",
    }
    meta = sample["meta"]
    assert meta["generator"] == "QwQ-32B" and meta["upstream_source"] == "nvidia/OpenCodeReasoning"
    assert meta["question_licence"] == "CC-BY-4.0" and "human-written" in meta["question_author"]
    for upstream in (
        "organic-chemistry-questions",
        "stackexchange_codegolf",
        "stackexchange-physics",
        "ai2-adapt-dev/openmath-2-math",
        None,
    ):
        assert cv.convert(ot_row(upstream), source, 100) == (None, cv.UPSTREAM)
    assert cv.convert(ot_row(answer="<think>" + "x" * 1000 + "</think>4"), source, 100) == (
        None,
        cv.LONG_THINKING,
    )
    assert cv.convert(ot_row(answer="<think>cut off"), source, 100) == (None, cv.NO_REASONING)


def test_range_slices_keep_only_whole_lines_and_files_are_read_in_turn():
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
    assert list(interleave([iter([1, 2, 3]), iter(["a"]), iter([])])) == [1, "a", 2, 3]


def chat_rows(prefix, n, subset=None, **meta):
    return [
        {
            "id": f"{prefix}-{i}",
            "messages": [
                {"role": "user", "content": f"task {prefix} {i}"},
                {"role": "assistant", "content": "done"},
            ],
            "meta": {"subset": subset or prefix, "licence": "L", **meta},
        }
        for i in range(n)
    ]


def graft_rows(prefix, n, subset="nemotron-v2-tool-calling", generator="GLM-4.6", **meta):
    """Rows as `graft fetch` writes them: a registered subset at its pinned revision."""
    source = SOURCES[subset]
    return chat_rows(
        prefix,
        n,
        subset=subset,
        source=source.dataset,
        revision=source.revision,
        licence=source.licence,
        generator=generator,
        **meta,
    )


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
    assert counts["nemotron"]["taken"] == round(2600 / 0.45 * 0.15)
    assert counts["openthoughts"]["taken"] == round(2600 / 0.45 * 0.10)
    assert plan_counts({"hosted": 450, "nemotron": 10})["nemotron"]["taken"] == 10
    with pytest.raises(ValueError, match="anchor"):
        plan_counts({"nemotron": 10})


def test_mix_is_deterministic_and_balances_subsets():
    ocr = {"upstream_source": "nvidia/OpenCodeReasoning", "question_licence": "CC-BY-4.0"}
    buckets = {
        "hosted": chat_rows("h", 90),
        "nemotron": graft_rows("a", 100)
        + graft_rows("b", 5, "nemotron-v1-tool-calling", "Qwen3-235B-A22B-Thinking-2507"),
        "openthoughts": graft_rows("o", 100, "openthoughts3", "QwQ-32B", **ocr),
    }
    one, report = mix(buckets, seed=3, heldout=None)
    two, _ = mix(buckets, seed=3, heldout=None)
    assert [r["id"] for r in one] == [r["id"] for r in two]
    assert report["buckets"]["nemotron"]["taken"] == 30
    assert report["by_source"]["nemotron-v1-tool-calling"] == 5  # not crowded out
    assert report["total"] == 90 + 30 + 20


def test_mix_rejects_unanswered_rows_other_models_records_and_heldout_overlap():
    class Heldout:
        def leaks(self, text):
            return "secret task" in text

    leaky = chat_rows("x", 1)
    leaky[0]["messages"][0]["content"] = "please do the secret task"
    unanswered = chat_rows("u", 1)
    unanswered[0]["messages"].pop()
    research = chat_rows("r", 4, for_model="emperor", judged=True)
    research += chat_rows("c", 3, for_model="consul", judged=True)
    research += chat_rows("n", 2, for_model="emperor")  # generated but not judged
    buckets = {
        "hosted": chat_rows("h", 45) + leaky,
        "research": research + [{"id": "dpo", "prompt": []}] + unanswered,
    }
    rows, report = mix(buckets, heldout=Heldout(), model="emperor")
    assert report["rejected"]["hosted"] == {"heldout_overlap": 1}
    assert report["rejected"]["research"] == {
        "for_consul": 3,
        "no_messages": 1,
        "not_answered": 1,
        "not_judged": 2,
    }
    assert {r["id"] for r in rows} >= {f"r-{i}" for i in range(4)}
    _, consul = mix(buckets, heldout=None, model="consul")
    assert consul["rejected"]["research"]["for_emperor"] == 6
    with pytest.raises(ValueError, match="Consul"):
        mix(buckets, model="consul")  # Emperor's held-out set is not Consul's


def test_stale_or_disallowed_grafts_are_rejected():
    stale = graft_rows("s", 2)
    for row in stale:
        row["meta"]["revision"] = "0" * 40
    old = chat_rows("old", 2, subset="openthoughts3")  # an old file: no revision at all
    gpt = graft_rows("g", 1, generator="GPT-OSS-120B")
    math = graft_rows(
        "m", 1, "openthoughts3", "QwQ-32B", upstream_source="ai2-adapt-dev/openmath-2-math"
    )
    excluded = graft_rows("x", 1, "nemotron-v2-customer-service", "DeepSeek-V3.2")
    buckets = {
        "hosted": chat_rows("h", 45),
        "nemotron": stale + gpt + excluded,
        "openthoughts": old + math,
    }
    rows, report = mix(buckets, heldout=None)
    assert len(rows) == 45
    assert report["rejected"]["nemotron"] == {
        "graft_stale_revision": 2,
        "graft_generator_not_allowed": 1,
        "graft_unknown_or_excluded_source": 1,
    }
    assert report["rejected"]["openthoughts"] == {
        "graft_stale_revision": 2,
        "graft_upstream_not_allowed": 1,
    }


def test_the_bundled_heldout_index_is_the_default():
    rows, report = mix({"hosted": chat_rows("h", 3)})
    assert report["heldout_checked"] is True and len(rows) == 3


def research_paper(**extra):
    return {
        "paper_id": "p1",
        "title": "t",
        "licence": "CC-BY-4.0",
        "url": "u",
        "problem": "p",
        "prior_work": "w",
        "actual_path": ["a", "b"],
        "meta": {"extractor": "deepseek-v4-pro"},
        **extra,
    }


def test_generators_run_on_deepseek_only_and_on_open_papers_only():
    for model in ("claude-opus", "gpt-5", "kimi-k3.1-pro"):
        with pytest.raises(generators.GeneratorError):
            generators.require_open_teacher(model)
    for licence in ("CC-BY-NC-4.0", "CC-BY-SA-4.0", "CC-BY-ND-4.0", "PMC-OA-commercial"):
        with pytest.raises(generators.GeneratorError, match="not one of"):
            generators.research_messages(research_paper(licence=licence), "missing_step")
    with pytest.raises(generators.GeneratorError, match="extracted by DeepSeek"):
        generators.research_messages(research_paper(meta={"extractor": "claude"}), "wrong_order")


def test_extraction_records_its_extractor():
    paper = {"paper_id": "p1", "title": "t", "licence": "CC0-1.0", "url": "u", "text": "body"}
    answer = {"content": json.dumps({"problem": "p", "prior_work": "w", "actual_path": ["a"]})}
    [row] = generators.extract_records(paper, answer, "deepseek-v4-pro")
    assert row["meta"]["extractor"] == "deepseek-v4-pro" and row["problem"] == "p"
    generators.research_messages(row, "missing_step")  # ready for the research generator


def test_research_records_are_chat_prefixes_routed_to_each_model():
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
    records = generators.research_records(
        research_paper(), "missing_step", answer, "deepseek-v4-pro"
    )
    assert [r["meta"]["for_model"] for r in records] == ["emperor", "consul"]
    assert records[0]["messages"] == [{"role": "user", "content": "plan it"}]
    assert records[1]["meta"]["paper"]["extractor"] == "deepseek-v4-pro"
    with pytest.raises(generators.GeneratorError, match="depends on"):
        generators.check_plan([dict(steps[1], depends_on=["s9"])])
    with pytest.raises(generators.GeneratorError, match="kind"):
        generators.research_records(research_paper(), "wrong_order", answer, "deepseek-v4-pro")
    solved = generators.solve_records(
        records[0], {"content": "plan", "reasoning_content": "think"}, "deepseek-v4-pro"
    )[0]
    assert solved["messages"][-1]["reasoning_content"] == "think"
    assert solved["meta"]["solver"] == "deepseek-v4-pro"


def turn(tool="ask_user", **arguments):
    return {"content": "", "tool_calls": [{"name": tool, "arguments": arguments or {"q": "?"}}]}


def dpo_answer(**overrides):
    data = {
        "hidden_constraint": "which colour scheme the brand wants",
        "discoverable_fact": "the test runner configured in pyproject",
        "ask": turn(question="Which brand colours should the theme use?"),
        "guess": turn("spawn_agent", brief="Make the theme dark blue."),
        "look_up": turn("spawn_agent", brief="Report the test runner set in pyproject.toml."),
        "needless_question": turn(question="Which test runner do you use?"),
        **overrides,
    }
    return {"content": "```json\n" + json.dumps(data) + "\n```"}


def test_dpo_pairs_are_assistant_tool_turns_and_need_a_judge():
    task = {"id": "t1", "prompt": "Theme the site and run the tests."}
    pairs = generators.ask_pairs(task, dpo_answer(), "deepseek-v4-pro")
    assert [p["kind"] for p in pairs] == ["dpo/ask_vs_guess", "dpo/look_up_vs_ask"]
    for pair in pairs:
        for side in ("chosen", "rejected"):
            assert pair[side]["role"] == "assistant" and pair[side]["tool_calls"]
        assert pair["meta"]["needs_judge"] and not pair["meta"]["judged"]
        assert "messages" not in pair  # not an SFT sample
    with pytest.raises(generators.GeneratorError, match="needless_question"):
        generators.ask_pairs(task, dpo_answer(needless_question=None), "deepseek-v4-pro")
    with pytest.raises(generators.GeneratorError, match="look_up"):
        generators.ask_pairs(task, dpo_answer(look_up={"content": "x"}), "deepseek-v4-pro")
    leaky = turn(question="Is the test runner configured in pyproject?")
    with pytest.raises(generators.GeneratorError, match="discoverable fact"):
        generators.ask_pairs(task, dpo_answer(ask=leaky), "deepseek-v4-pro")
    for key, bad in (
        ("guess", turn("edit_file", path="x")),  # not one of Emperor's tools
        ("look_up", turn("ask_user", question="?")),  # looking up is not asking
        ("ask", turn("spawn_agent", brief="b")),  # asking is ask_user
    ):
        with pytest.raises(generators.GeneratorError, match="may use"):
            generators.ask_pairs(task, dpo_answer(**{key: bad}), "deepseek-v4-pro")
    judged = generators.apply_dpo_judgement(
        pairs[0], {"content": '{"valid": true}'}, "kimi-k3.1-pro"
    )
    assert judged["meta"]["judged"] and judged["meta"]["judge"] == "kimi-k3.1-pro"
    assert generators.apply_dpo_judgement(pairs[0], {"content": '{"valid": false}'}, "k") is None
    assert "chosen" in generators.dpo_judge_messages(pairs[0])[1]["content"]


def test_coref_dialogues_become_chat_prefixes_after_the_summary():
    answer = {
        "content": json.dumps(
            {
                "turns": [
                    {"role": "user", "content": "fix a.py"},
                    {"role": "assistant", "content": "done"},
                    {"role": "summary", "content": "a.py was fixed"},
                    {"role": "user", "content": "now the other one"},
                    {"role": "assistant", "content": "b.py?"},
                    {"role": "user", "content": "yes, and its tests"},
                ],
                "resolutions": [{"turn": 3, "refers_to": "b.py"}],
            }
        )
    }
    [record] = generators.coref_records({"id": "s1"}, answer, "deepseek-v4-pro")
    first = record["messages"][0]
    assert first["role"] == "system" and "a.py was fixed" in first["content"]
    assert record["messages"][-1] == {"role": "user", "content": "yes, and its tests"}


def test_run_resumes_skips_heldout_papers_and_survives_teacher_errors():
    rows = [
        {"id": "t1", "prompt": "do x"},
        {"id": "t2", "prompt": "do y"},
        {"id": "t3", "prompt": "z"},
    ]

    def chat(messages):
        prompt = messages[-1]["content"]
        if prompt == "do y":
            return {"content": "no json here"}
        if prompt == "z":
            raise TeacherOutputError("server could not parse the output")
        return dpo_answer()

    out = list(generators.run("ask-dpo", rows, chat, "deepseek-v4-pro"))
    made = [r for r, e in out if r]
    errors = [e for r, e in out if e]
    assert [r["kind"] for r in made] == ["dpo/ask_vs_guess", "dpo/look_up_vs_ask"]
    assert [e.split(":")[0] for e in errors] == ["t2", "t3"]
    every = {"t1", "t2", "t3"}
    assert list(generators.run("ask-dpo", rows, chat, "deepseek-v4-pro", done=every)) == []
    held = list(
        generators.run(
            "research", [research_paper()], chat, "deepseek-v4-pro", heldout_papers={"p1"}
        )
    )
    assert held == [(None, "p1: paper p1 is held out for the eval split")]
    with pytest.raises(generators.GeneratorError, match="held-out paper list"):
        list(generators.run("research", [research_paper()], chat, "deepseek-v4-pro"))

    class Heldout:
        def leaks(self, text):
            return "do x" in text

    out = list(
        generators.run("ask-dpo", rows[:1], chat, "deepseek-v4-pro", heldout_tasks=Heldout())
    )
    assert out == [(None, "t1: overlaps a held-out eval task")]


def test_solve_takes_research_records_only():
    with pytest.raises(generators.GeneratorError, match="research records only"):
        generators.solve_messages(
            {"kind": "paraphrase", "messages": [{"role": "user", "content": "x"}]}
        )


def test_allow_list_is_anchored():
    assert not allowed("DeepSeek-R1-Distill-Llama-70B")
    assert not allowed("deepseek-v3-but-actually-gpt")
    for model in ("deepseek-v4-pro", "deepseek-flash"):
        assert generators.require_open_teacher(model) == model
    with pytest.raises(generators.GeneratorError):
        generators.require_open_teacher("deepseek-r1-distill-llama-70b")


def test_generate_refuses_without_confirm_spend_and_shares_the_hosted_ledger(tmp_path, capsys):
    code = graft_main(["generate", "coref", "--input", str(tmp_path / "x"), "--out", str(tmp_path)])
    assert code == 2 and "--confirm-spend" in capsys.readouterr().err
    assert DEFAULT_LEDGER.as_posix() == "data/hosted-1/spend.jsonl"


def test_mix_cli_writes_the_mix_and_report(tmp_path):
    hosted = tmp_path / "hosted.jsonl"
    hosted.write_text("".join(json.dumps(s) + "\n" for s in chat_rows("h", 45)), encoding="utf-8")
    graft = tmp_path / "graft"
    graft.mkdir()
    ocr = graft_rows(
        "o", 50, "openthoughts3", "QwQ-32B", upstream_source="nvidia/OpenCodeReasoning"
    )
    (graft / "openthoughts3.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in ocr), encoding="utf-8"
    )
    out = tmp_path / "mix"
    args = ["mix", "--model", "emperor", "--hosted", str(hosted), "--graft", str(graft)]
    assert graft_main([*args, "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["buckets"]["openthoughts"]["taken"] == 10  # 45:10 with the others missing
    assert report["buckets"]["nemotron"]["missing"] and report["model"] == "emperor"
    assert len((out / "final.jsonl").read_text(encoding="utf-8").splitlines()) == 55


def test_the_card_credits_grafts_and_does_not_call_them_judged():
    graft = graft_rows("g", 2)
    ocr = graft_rows(
        "o",
        1,
        "openthoughts3",
        "QwQ-32B",
        upstream_source="nvidia/OpenCodeReasoning",
        question_licence="CC-BY-4.0",
    )
    buckets = {"hosted": chat_rows("h", 45), "nemotron": graft, "openthoughts": ocr}
    _, report = mix(buckets, heldout=None)
    assert report["grafts"][0] == {
        "dataset": "nvidia/Nemotron-SFT-Agentic-v2",
        "subset": "nemotron-v2-tool-calling",
        "revision": SOURCES["nemotron-v2-tool-calling"].revision,
        "licence": "CC-BY-4.0",
        "generators": ["GLM-4.6"],
        "question_licences": [],
        "samples": 2,
    }
    assert report["grafts"][1]["question_licences"] == ["nvidia/OpenCodeReasoning: CC-BY-4.0"]
    text = card.render_card(card.CardInputs(config=parse_config({}), mix_report=report), "bf16")
    assert "Filters (our hosted samples)" in text and "**not** judged" in text
    assert "nvidia/Nemotron-SFT-Agentic-v2 (nemotron-v2-tool-calling" in text
    assert "creativecommons.org/licenses/by/4.0" in text
    assert "questions nvidia/OpenCodeReasoning: CC-BY-4.0" in text
    plain = card.render_card(card.CardInputs(config=parse_config({})), "bf16")
    assert "grafts" not in plain.lower()
