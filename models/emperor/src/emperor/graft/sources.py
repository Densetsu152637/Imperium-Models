"""The open datasets we graft from, with their licences and generators as their dataset cards
state them (checked 2026-11-15), and the rule that drops samples made by closed models.

A subset is usable only when every sample's generator is known: either the sample names it
(`generator_field`), or the card names one set of models for the whole subset. A subset whose
card names a blocked model without saying which samples it made, or names no generator at all,
is excluded as a whole.
"""

from dataclasses import dataclass, field

# Generators whose outputs we do not train on: GPT models (including the open-weight gpt-oss,
# by the user's rule "no GPT"), Claude and Gemini. Matched case-insensitively as substrings.
BLOCKED_GENERATORS = ("gpt", "claude", "gemini", "openai", "anthropic", "google/")


def blocked(generator: str | None) -> bool:
    """True when a generator is unknown or made by a blocked model family."""
    if not generator or not generator.strip():
        return True
    name = generator.lower()
    return any(word in name for word in BLOCKED_GENERATORS)


@dataclass(frozen=True)
class Source:
    key: str  # short name used in sample ids and reports
    dataset: str  # Hugging Face dataset id
    files: tuple[str, ...]  # files in the dataset repo
    licence: str
    kind: str  # "nemotron" (OpenAI-style messages and tools) or "openthoughts"
    card_generators: tuple[str, ...] = ()  # the generators the card names for the subset
    generator_field: str = ""  # dotted path to a per-sample generator, when the data has one
    note: str = ""
    excluded_reason: str = ""  # set when the whole subset is unusable
    url: str = field(default="")

    def __post_init__(self):
        if not self.url:
            object.__setattr__(self, "url", f"https://huggingface.co/datasets/{self.dataset}")

    @property
    def usable(self) -> bool:
        return not self.excluded_reason

    def generator_of(self, record: dict) -> str | None:
        """The model that made this sample: its own field, else the card's single generator
        set (joined with '+'); None when it cannot be known."""
        if self.generator_field:
            value = record
            for part in self.generator_field.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None
        return "+".join(self.card_generators) or None


NEMOTRON_V1 = "nvidia/Nemotron-Agentic-v1"
NEMOTRON_V2 = "nvidia/Nemotron-SFT-Agentic-v2"
OPENTHOUGHTS = "open-thoughts/OpenThoughts3-1.2M"

SOURCES: dict[str, Source] = {
    s.key: s
    for s in (
        Source(
            key="nemotron-v1-tool-calling",
            dataset=NEMOTRON_V1,
            files=("data/tool_calling.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            card_generators=("Qwen3-235B-A22B-Thinking-2507", "Qwen3-235B-A22B-Instruct-2507"),
            note="card: simulated and judged with Qwen3-235B-A22B Thinking/Instruct-2507",
        ),
        Source(
            key="nemotron-v1-interactive-agent",
            dataset=NEMOTRON_V1,
            files=("data/interactive_agent.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            card_generators=(
                "Qwen3-235B-A22B-Thinking-2507",
                "Qwen3-32B",
                "GPT-OSS-120B",
                "Qwen3-235B-A22B-Instruct-2507",
            ),
            excluded_reason=(
                "card lists GPT-OSS-120B among the generators and the samples do not say which "
                "model made them"
            ),
        ),
        Source(
            key="nemotron-v2-tool-calling",
            dataset=NEMOTRON_V2,
            files=("data/tool_calling.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            generator_field="metadata.sdg_model",
            note="card: DeepSeek-V3.2 and GLM-4.6 (plus Toucan samples); per-sample sdg_model",
        ),
        Source(
            key="nemotron-v2-customer-service",
            dataset=NEMOTRON_V2,
            files=("data/interactive_agent.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            generator_field="model",
            note="card names no generator; each sample names its model",
        ),
        Source(
            key="nemotron-v2-search",
            dataset=NEMOTRON_V2,
            files=("data/search.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            excluded_reason="the card and the samples name no generator model",
        ),
        Source(
            key="openthoughts3",
            dataset=OPENTHOUGHTS,
            # One shard from each domain block (code, math, science) of the 120.
            files=(
                "data/train-00000-of-00120.parquet",
                "data/train-00060-of-00120.parquet",
                "data/train-00115-of-00120.parquet",
            ),
            licence="Apache-2.0",
            kind="openthoughts",
            card_generators=("QwQ-32B",),
            note="card: annotations by QwQ-32B",
        ),
    )
}

# Which mix bucket each source feeds.
BUCKETS = {
    "nemotron": (
        "nemotron-v1-tool-calling",
        "nemotron-v2-tool-calling",
        "nemotron-v2-customer-service",
    ),
    "openthoughts": ("openthoughts3",),
}
