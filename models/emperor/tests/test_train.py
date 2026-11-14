import json
from pathlib import Path

import pytest
from conftest import make_sample

from emperor.export.gguf import find_converter
from emperor.train.cli import main as train_main
from emperor.train.config import ConfigError, load_config, parse_config
from emperor.train.data import (
    IGNORE,
    NO_REASONING,
    TOO_LONG,
    MissingReasoning,
    assistant_spans,
    chat_messages,
    encode_sample,
    labels_from_offsets,
    last_query_index,
    pad_batch,
    reasoning_spans,
    renders_thinking,
    split_records,
)

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_config_loads_and_is_bf16_lora():
    config = load_config(ROOT / "configs" / "sft.toml")
    assert config.dtype == "bfloat16"
    assert config.data.max_length == 8192
    assert config.model.base.endswith("Qwen3.5-4B")
    assert config.merge.output_dir == "weights/emperor-4b-8k-bf16"
    assert config.train.save_steps > 0 and config.train.gradient_checkpointing


def test_defaults_parse_from_an_empty_file():
    assert parse_config({}).lora.r == 32


@pytest.mark.parametrize("key", ["load_in_4bit", "load_in_8bit", "qlora", "bnb_4bit_quant_type"])
def test_quantised_training_is_refused(key):
    with pytest.raises(ConfigError, match="not supported"):
        parse_config({"model": {key: True}})


def test_unknown_keys_and_bad_values_are_refused():
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"train": {"epochz": 1}})
    with pytest.raises(ConfigError, match="unknown section"):
        parse_config({"trainer": {}})
    with pytest.raises(ConfigError):
        parse_config({"data": {"max_length": 10}})
    with pytest.raises(ConfigError):
        parse_config({"data": {"val_fraction": 0.9}})


def test_config_file_errors_are_reported(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[model\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
    assert train_main(["show", "--config", str(bad)]) == 2
    assert train_main(["show", "--config", str(ROOT / "configs" / "sft.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["lora"]["r"] == 32


def test_split_is_deterministic_and_stable_under_additions():
    records = [{"id": f"p-{i}"} for i in range(400)]
    train, val = split_records(records, 0.1, seed=1)
    assert len(train) + len(val) == 400 and 15 <= len(val) <= 70
    again = split_records(records, 0.1, seed=1)
    assert (train, val) == again
    more = split_records(records + [{"id": "p-new"}], 0.1, seed=1)
    assert {r["id"] for r in val} <= {r["id"] for r in more[1]}
    assert split_records(records, 0.0, seed=1)[1] == []


def test_tool_call_arguments_become_objects_for_the_chat_template():
    messages = chat_messages(make_sample())
    arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, dict) and arguments["tier"] == "standard"


def fake_render(messages, add_generation_prompt):
    text = ""
    for message in messages:
        text += f"<|{message['role']}|>{message.get('content') or ''}<|end|>\n"
    if add_generation_prompt:
        text += "<|assistant|>"
    return text


def test_assistant_spans_cover_only_assistant_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "bye"},
    ]
    # Template whose generation prompt is "<|assistant|>" and whose turns open the same way.
    full, spans = assistant_spans(messages, fake_render)
    assert [full[a:b] for a, b in spans] == ["hello<|end|>", "bye<|end|>"]


def test_labels_mask_everything_outside_assistant_spans():
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    ids = [10, 11, 12, 13]
    assert labels_from_offsets(offsets, ids, [(6, 12)]) == [IGNORE, IGNORE, 12, 13]
    # A token that straddles the boundary is not trained on.
    assert labels_from_offsets(offsets, ids, [(7, 12)]) == [IGNORE, IGNORE, IGNORE, 13]


def test_padding_masks_labels_and_attention():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4], "labels": [4], "attention_mask": [1]},
    ]
    out = pad_batch(batch, pad_token_id=0)
    assert out["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert out["labels"] == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


def test_gguf_export_is_optional(tmp_path, monkeypatch):
    monkeypatch.delenv("EMPEROR_LLAMA_CPP", raising=False)
    assert find_converter() is None
    (tmp_path / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    assert find_converter(tmp_path) == tmp_path / "convert_hf_to_gguf.py"


def test_unpadded_batches_drop_the_attention_mask():
    batch = pad_batch([{"input_ids": [1, 2], "labels": [IGNORE, 2], "attention_mask": [1, 1]}], 0)
    assert "attention_mask" not in batch and batch["position_ids"] == [[0, 1]]


def test_training_stops_when_vram_is_short(monkeypatch):
    import sys
    import types

    from emperor.train.sft import require_free_vram

    cuda = types.SimpleNamespace(mem_get_info=lambda device: (8 * 2**30, 16 * 2**30))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    with pytest.raises(SystemExit, match="only 8.0 GiB of VRAM is free"):
        require_free_vram(13.5)
    require_free_vram(6.0)


def thinking_render(messages, add_generation_prompt):
    """The assistant logic of Qwen3 thinking templates (Thinking-2507; Qwen3.5 differs only in
    always writing the block after the last query), without the tool section."""
    query = last_query_index(messages)
    text = ""
    for i, message in enumerate(messages):
        content = message.get("content") or ""
        if message["role"] != "assistant":
            text += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
            continue
        reasoning = message.get("reasoning_content") or ""
        if i > query:  # Qwen3.5 writes the block, empty or not, on every turn of the loop
            text += f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content}"
        else:
            text += f"<|im_start|>assistant\n{content}"
        for n, call in enumerate(message.get("tool_calls") or []):
            text += "\n" if n or content else ""
            text += f"<tool_call>{call['function']['name']}</tool_call>"
        text += "<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n<think>\n"
    return text


def thinking_messages(with_reasoning=True):
    messages = chat_messages(make_sample())
    for i, message in enumerate(messages):
        if message["role"] == "assistant" and with_reasoning:
            message["reasoning_content"] = f"thought {i}"
    return messages


def test_a_thinking_template_is_recognised():
    assert renders_thinking(thinking_render)
    assert not renders_thinking(fake_render)


def test_reasoning_spans_train_the_reasoning_and_answer_of_every_turn():
    full, spans = reasoning_spans(thinking_messages(), thinking_render)
    trained = [full[a:b] for a, b in spans]
    assert trained == [
        "thought 2\n</think>\n\n<tool_call>spawn_agent</tool_call><|im_end|>",
        "thought 4\n</think>\n\n<tool_call>await_tasks</tool_call><|im_end|>",
        "thought 6\n</think>\n\nPagination is added; tests pass.<|im_end|>",
    ]
    # The `<think>` opener comes from the generation prompt, so it is context, not a label.
    assert all(full[a - len("<think>\n") : a] == "<think>\n" for a, _ in spans)


def test_only_the_first_turn_must_have_reasoning():
    messages = thinking_messages()
    del messages[2]["reasoning_content"]
    with pytest.raises(MissingReasoning):
        reasoning_spans(messages, thinking_render)
    with pytest.raises(MissingReasoning):
        reasoning_spans(thinking_messages(with_reasoning=False), thinking_render)
    # A later turn without reasoning is trained with an empty think block.
    messages = thinking_messages()
    del messages[4]["reasoning_content"]
    full, spans = reasoning_spans(messages, thinking_render)
    assert (
        full[spans[1][0] : spans[1][1]]
        == "\n</think>\n\n<tool_call>await_tasks</tool_call><|im_end|>"
    )


def test_turns_before_the_last_user_message_are_context_only():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},  # no reasoning: rendered without it anyway
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "bye", "reasoning_content": "wrap up"},
    ]
    full, spans = reasoning_spans(messages, thinking_render)
    assert [full[a:b] for a, b in spans] == ["wrap up\n</think>\n\nbye<|im_end|>"]
    assert last_query_index(messages) == 3


class CharTokenizer:
    """One token per character, with the template above: enough to run `encode_sample`."""

    def __init__(self, render):
        self.render = render

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        return self.render(messages, add_generation_prompt)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_encode_sample_puts_reasoning_in_the_labels_and_skips_samples_without_it():
    tokenizer = CharTokenizer(thinking_render)
    sample = make_sample()
    for message in sample["messages"]:
        if message["role"] == "assistant":
            message["reasoning_content"] = "I should delegate this."
    row, reason = encode_sample(sample, tokenizer, 100_000)
    assert reason is None
    labelled = "".join(chr(t) for t in row["labels"] if t != IGNORE)
    assert labelled.count("I should delegate this.\n</think>") == 3
    assert "Pagination is added" in labelled and "You are Emperor" not in labelled

    assert encode_sample(make_sample(), tokenizer, 100_000) == (None, NO_REASONING)
    assert encode_sample(sample, tokenizer, 50) == (None, TOO_LONG)
    # A non-thinking base trains reasoning-less samples as before.
    row, reason = encode_sample(make_sample(), CharTokenizer(fake_render), 100_000)
    assert reason is None and any(t != IGNORE for t in row["labels"])


def test_fp32_base_tensors_are_found_under_text_model_names(tmp_path):
    from emperor.train.sft import fp32_tensors, text_name

    header = {
        "__metadata__": {"format": "pt"},
        "model.language_model.layers.0.linear_attn.A_log": {
            "dtype": "F32",
            "shape": [32],
            "data_offsets": [0, 128],
        },
        "model.language_model.layers.0.mlp.up_proj.weight": {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [128, 130],
        },
        "mtp.norm.weight": {"dtype": "F32", "shape": [1], "data_offsets": [130, 134]},
        "model.visual.blocks.0.norm1.bias": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [134, 138],
        },
    }
    raw = json.dumps(header).encode()
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(138))
    found = fp32_tensors(tmp_path)
    assert found == {
        "model.layers.0.linear_attn.A_log": (
            shard,
            "model.language_model.layers.0.linear_attn.A_log",
        )
    }
    assert text_name("model.layers.1.mlp.up_proj.weight") == "model.layers.1.mlp.up_proj.weight"


def test_offloading_is_opt_in_and_wraps_the_training_step():
    from contextlib import contextmanager

    from emperor.train.sft import offloading_trainer

    assert parse_config({}).train.offload_activations is False
    assert parse_config({"train": {"offload_activations": True}}).train.offload_activations

    entered = []

    class Graph:
        @staticmethod
        @contextmanager
        def save_on_cpu(pin_memory):
            entered.append(pin_memory)
            yield

    class Torch:
        class autograd:
            graph = Graph

        class cuda:
            @staticmethod
            def empty_cache():
                entered.append("emptied")

    class Base:
        def training_step(self, *args, **kwargs):
            return "loss"

    trainer = offloading_trainer(Base, Torch)()
    assert trainer.training_step() == "loss" and entered == ["emptied", True]


def test_the_training_length_covers_the_p99_of_the_data_up_to_a_cap():
    from emperor.train.data import pick_max_length

    lengths = list(range(1000, 11000, 100))  # 100 samples, 1,000 to 10,900 tokens
    assert pick_max_length(lengths) == 11264  # p99 is 10,800, rounded up to 1,024s
    assert pick_max_length(lengths, cap=8192) == 8192
    assert pick_max_length([40000] * 10) == 32768  # a runaway length is capped
    assert pick_max_length([]) == 1024
    assert pick_max_length(lengths, quantile=1.0) == 11264  # the longest, 10,900
    config = parse_config({"data": {"max_length_auto": True}})
    assert config.data.max_length_cap == 32768 and config.data.max_length_quantile == 1.0


@pytest.mark.parametrize(
    ("path", "name", "base"),
    [
        ("configs/emperor-4b.toml", "emperor-4b", "4B"),
        ("configs/emperor-lite-2b.toml", "emperor-lite-2b", "2B"),
    ],
)
def test_the_emperor_configs(path, name, base):
    config = load_config(ROOT / path)
    assert config.model.base.endswith(f"Qwen3.5-{base}")
    assert config.release.name == name and config.release.size == base
    assert config.release.base_model == f"Qwen/Qwen3.5-{base}"
    assert config.train.output_dir == f"outputs/{name}"
    assert config.merge.output_dir == f"weights/{name}-bf16"
    assert config.merge.adapter_dir.startswith(config.train.output_dir)
    # Training length from the data, never above 16K; longer samples are dropped.
    assert config.data.max_length_auto and config.data.max_length_cap == 16384
    assert config.train.offload_activations and config.train.vram_cap_gib == 15.3
    # Just over the 16K probe's 14.03 GiB peak: a model left resident blocks the start.
    assert 14.03 < config.train.offload_min_free_vram_gib <= 14.5


# Every config of both packages, by model folder. Outputs are relative to the model folder.
EVERY_CONFIG = [
    ("emperor", "configs/emperor-4b.toml"),
    ("emperor", "configs/emperor-lite-2b.toml"),
    ("emperor", "configs/sft.toml"),
    ("emperor", "configs/sft-r2.toml"),
    ("consul", "../consul/configs/consul-4b.toml"),
    ("consul", "../consul/configs/consul-lite-2b.toml"),
    ("consul", "../consul/configs/sft.toml"),
]


def test_no_two_configs_write_to_the_same_place():
    configs = [(folder, load_config(ROOT / path)) for folder, path in EVERY_CONFIG]
    names = [c.release.name for _, c in configs]
    assert len(set(names)) == len(names)
    for attribute in ("train", "merge"):
        places = [(folder, getattr(c, attribute).output_dir) for folder, c in configs]
        assert len(set(places)) == len(places), attribute
    # The CLIs default to configs/sft.toml: it must not share the new 4B configs' names.
    for folder, path in (("emperor", "configs/sft.toml"), ("consul", "../consul/configs/sft.toml")):
        assert load_config(ROOT / path).release.name.endswith("-8k"), folder


def test_the_four_models_have_distinct_aliases_and_text():
    models = [load_config(ROOT / p) for _, p in EVERY_CONFIG if "sft" not in p]
    assert len(models) == 4
    assert len({c.release.eval_alias for c in models}) == 4
    assert len({c.release.summary for c in models}) == 4


@pytest.mark.parametrize("size", ["2B", "4B"])
def test_the_kv_layout_matches_the_base_config(size):
    from emperor.release import card

    base = Path(f"F:/AI-Models/Qwen/Qwen3.5-{size}")
    if not (base / "config.json").is_file():
        pytest.skip(f"no local base at {base}")
    name = "emperor-4b" if size == "4B" else "emperor-lite-2b"
    release = load_config(ROOT / "configs" / f"{name}.toml").release
    assert card.config_arch(release) == card.read_arch(base)


def test_vram_cap_is_clamped_to_the_free_vram():
    from emperor.train.sft import vram_fraction

    gib = 2**30
    total = 16 * gib
    # A fixed cap is clamped to the VRAM free now, less 0.2 GiB.
    assert vram_fraction(15.6 * gib, total, cap_gib=15.3) == pytest.approx(15.3 / 16)
    assert vram_fraction(10 * gib, total, cap_gib=15.3) == pytest.approx(9.8 / 16)
    assert vram_fraction(14.2 * gib, total, cap_gib=15.3) == pytest.approx(14.0 / 16)
    assert vram_fraction(4 * gib, total, cap_gib=20.0) == pytest.approx(3.8 / 16)
    assert vram_fraction(12.5 * gib, total) == pytest.approx(12 / 16)  # free less 0.5 GiB
    assert vram_fraction(0, total) == 0.1


def test_training_checks_free_vram_then_caps_it_from_the_config(monkeypatch):
    import inspect

    from emperor.train import sft

    calls = []
    monkeypatch.setattr(sft, "require_free_vram", lambda gib: calls.append(("min", gib)))
    monkeypatch.setattr(sft, "cap_vram", lambda gib: calls.append(("cap", gib)) or gib)
    config = load_config(ROOT / "configs" / "emperor-4b.toml")
    assert sft.prepare_vram(config) == 15.3
    assert calls == [("min", 14.1), ("cap", 15.3)]
    calls.clear()
    plain = parse_config({"train": {"vram_cap_gib": 12.0}})  # no offload: the model's floor
    sft.prepare_vram(plain)
    assert calls == [("min", plain.model.min_free_vram_gib), ("cap", 12.0)]
    # train() sets the VRAM up through prepare_vram, before loading anything.
    source = inspect.getsource(sft.train)
    assert "prepare_vram(config)" in source
    assert source.index("prepare_vram(config)") < source.index("from_pretrained")


def test_vram_settings_and_length_keys_are_validated():
    assert parse_config({"data": {"max_length_quantile": 0.99}}).data.max_length_quantile == 0.99
    with pytest.raises(ConfigError, match="not supported"):
        parse_config({"data": {"quantile_bits": 4}})
    with pytest.raises(ConfigError, match="vram_cap_gib"):
        parse_config({"train": {"vram_cap_gib": -1}})
    with pytest.raises(ConfigError, match="minimum free VRAM"):
        parse_config({"train": {"offload_min_free_vram_gib": -1}})
    with pytest.raises(ConfigError, match="max_length_quantile"):
        parse_config({"data": {"max_length_quantile": 0}})
    with pytest.raises(ConfigError, match="KV layout"):
        parse_config({"release": {"kv_heads": 0}})
