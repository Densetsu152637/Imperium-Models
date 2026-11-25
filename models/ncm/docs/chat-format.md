# NCM chat format

The text format of every fine-tuning sample for the NCM models (Emperor and Consul). The stage 2 tokenizer reserves each tag below as a single special token, so no tag may appear inside message content. One renderer, `ncm_model::chat` (`crates/ncm-model/src/chat.rs`), produces it for every crate: `ncm-sft` renders the token-era teacher data and refuses records whose content already holds a reserved tag, `ncm-data` renders the pretraining chat and reasoning sources and defuses reserved tags in them (a space after the first character, `< |im_end|>`), and the runtime renders prompts.

## Turns

```text
<|im_start|>{role}\n{content}<|im_end|>\n
```

Roles are `system`, `user`, `assistant` and `tool`. Content is trimmed of surrounding whitespace and ends with a newline before `<|im_end|>`.

## System turn and tools

The system turn holds the system prompt, then the tool definitions the sample was written against, one compact JSON object per line (`name`, `description`, `parameters`). With tools and no system prompt the turn holds only the block. Without tools there is no block.

```text
<|im_start|>system
You are Emperor, the root agent of an Imperium session. ...

<tools>
{"name":"web_search","description":"...","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}
{"name":"await_tasks","description":"...","parameters":{...}}
</tools>
<|im_end|>
```

The definitions are Imperium's tool contract (`crates/ncm-sft/tools/`). Qwen's `# Tools` boilerplate and default system prompt are not used.

## Assistant turn

Parts appear in this order, joined by a newline (the think block is always present, the rest only when present):

1. `<think>\n{reasoning}\n</think>`: always present, so every assistant turn starts with `<think>` and the model learns to choose between thinking and closing the block at once. A turn without reasoning (the teacher often acts on a tool result without thinking again) gets an empty block, `<think>\n\n</think>`. Earlier turns keep their reasoning in the history.
2. The visible text.
3. One `<tool_call>\n{json}\n</tool_call>` block per call, in order. The JSON is `{"name": ..., "arguments": {...}}` with the arguments as an object, keys in the teacher's order.

The span from the first character of the body through `<|im_end|>` is what SFT trains on; `ncm-sft` writes it as `assistant_spans` (byte ranges of `text`).

```text
<|im_start|>assistant
<think>
Search first, then decide.
</think>
Checking two sources.
<tool_call>
{"name":"web_search","arguments":{"query":"Dramatiq vs Celery"}}
</tool_call>
<tool_call>
{"name":"web_search","arguments":{"query":"Celery memory prefork"}}
</tool_call>
<|im_end|>
```

## Tool turn

Each result is its own `tool` turn wrapped in `<tool_response>`. Results follow the order of the calls they answer (the structured `messages` keep `tool_call_id`).

```text
<|im_start|>tool
<tool_response>
Dramatiq vs Celery: which Python task queue ...
</tool_response>
<|im_end|>
```

## Plain conversation

```text
<|im_start|>system
You are Consul. ...
<|im_end|>
<|im_start|>user
Review this plan: ...
<|im_end|>
<|im_start|>assistant
<think>
The migration step has no rollback.
</think>
{"verdict":"revise","findings":[...]}
<|im_end|>
```

## Dataset files

`ncm-sft build` writes, per model, `sft.jsonl`, `needs-regeneration.jsonl` and `stats.json` under `data/sft/<model>/`. Each `sft.jsonl` line has `id`, `source_file`, `split` (`train` or `val`, about 2% val by SHA-256 of the id), `meta`, `tools` (definitions), `messages` (structured), `text` (this format) and `assistant_spans`.

Structured `messages`: `{role, content, reasoning?, tool_calls?: [{id, name, arguments}], tool_call_id?}`.
