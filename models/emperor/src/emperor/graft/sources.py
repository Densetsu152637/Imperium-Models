"""The open datasets we graft from, with their licences and generators as their dataset cards
state them (checked 2026-11-15), pinned to the dataset revisions that were checked.

Generators are an allow-list: a sample is used only when the model that made it (its own field,
or the one set of models the card names for the subset) is an explicitly named open model. A
subset whose card names any other model without saying which samples it made, or names no
generator at all, is excluded as a whole; so is a subset whose user turns or tool outputs come
from an unnamed model, since they stay in the training context even when masked.

OpenThoughts also carries its questions' upstream sources; only allow-listed upstream sources
(with a known licence and a human or allowed-model author) are used.
"""

import re
from dataclasses import dataclass, field

# Open models whose outputs may be trained on (matched against the normalised name, without an
# organisation prefix). Everything else, including every GPT/gpt-oss, Claude and Gemini model,
# and any unnamed model, is dropped.
ALLOWED_GENERATORS = (
    re.compile(r"^deepseek-(v3(\.\d+)?|r1|v4)", re.IGNORECASE),
    re.compile(r"^qwen3", re.IGNORECASE),
    re.compile(r"^qwq", re.IGNORECASE),
    re.compile(r"^glm-4(\.\d+)?", re.IGNORECASE),
)


def normalise(generator: str) -> str:
    return generator.strip().rsplit("/", 1)[-1]


def allowed(generator: str | None) -> bool:
    """True when every model named (joined with '+') is on the allow-list."""
    if not generator or not generator.strip():
        return False
    names = [normalise(name) for name in generator.split("+")]
    return all(any(p.match(name) for p in ALLOWED_GENERATORS) for name in names)


@dataclass(frozen=True)
class Upstream:
    """Where a dataset's questions came from: their licence and who wrote them."""

    licence: str
    author: str


# OpenThoughts3 question sources (arXiv 2506.04178). Allowed:
OPENTHOUGHTS_UPSTREAM = {
    "nvidia/OpenCodeReasoning": Upstream(
        "CC-BY-4.0",
        "human-written competitive programming problems (TACO, APPS, CodeContests, Codeforces) "
        "collected in nvidia/OpenCodeReasoning",
    ),
}
# ...and why the others are not:
OPENTHOUGHTS_EXCLUDED_UPSTREAM = {
    "organic-chemistry-questions": "questions written by GPT-4o-mini",
    "stackexchange_codegolf": "CC-BY-SA (share-alike)",
    "stackexchange-physics": "CC-BY-SA (share-alike)",
    "ai2-adapt-dev/openmath-2-math": (
        "OpenMathInstruct-2 questions by Llama-3.1-405B; the Llama 3.1 licence requires derived "
        "model names to begin with 'Llama'"
    ),
}


@dataclass(frozen=True)
class Source:
    key: str  # short name used in sample ids and reports
    dataset: str  # Hugging Face dataset id
    revision: str  # the dataset commit whose card and files were checked
    files: tuple[str, ...]  # files in the dataset repo
    licence: str
    kind: str  # "nemotron" (OpenAI-style messages and tools) or "openthoughts"
    card_generators: tuple[str, ...] = ()  # the generators the card names for the subset
    generator_field: str = ""  # dotted path to a per-sample generator, when the data has one
    upstream_field: str = ""  # dotted path to the sample's upstream source, copied into meta
    upstream_allow: dict = field(default_factory=dict)  # upstream source -> Upstream; {} = any
    note: str = ""
    excluded_reason: str = ""  # set when the whole subset is unusable

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset}/tree/{self.revision}"

    @property
    def usable(self) -> bool:
        return not self.excluded_reason

    def _field(self, record: dict, path: str) -> str | None:
        value = record
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else None

    def generator_of(self, record: dict) -> str | None:
        """The model that made this sample: its own field, else the card's single generator
        set (joined with '+'); None when it cannot be known."""
        if self.generator_field:
            return self._field(record, self.generator_field)
        return "+".join(self.card_generators) or None

    def upstream_of(self, record: dict) -> str | None:
        return self._field(record, self.upstream_field) if self.upstream_field else None


NEMOTRON_V1 = "nvidia/Nemotron-Agentic-v1"
NEMOTRON_V1_REV = "650d590978ca35c8f1ecea2faf136e5fac421b62"
NEMOTRON_V2 = "nvidia/Nemotron-SFT-Agentic-v2"
NEMOTRON_V2_REV = "7c804833427f633ccd53b582dbf02525fd680f78"
OPENTHOUGHTS = "open-thoughts/OpenThoughts3-1.2M"
OPENTHOUGHTS_REV = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"

SOURCES: dict[str, Source] = {
    s.key: s
    for s in (
        Source(
            key="nemotron-v1-tool-calling",
            dataset=NEMOTRON_V1,
            revision=NEMOTRON_V1_REV,
            files=("data/tool_calling.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            card_generators=("Qwen3-235B-A22B-Thinking-2507", "Qwen3-235B-A22B-Instruct-2507"),
            note="card: user, agent and tool roles simulated and judged with Qwen3-235B-A22B",
        ),
        Source(
            key="nemotron-v1-interactive-agent",
            dataset=NEMOTRON_V1,
            revision=NEMOTRON_V1_REV,
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
            revision=NEMOTRON_V2_REV,
            files=("data/tool_calling.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            generator_field="metadata.sdg_model",
            upstream_field="metadata.source",
            note=(
                "card: all three roles simulated and judged with DeepSeek-V3.2 and GLM-4.6 "
                "(plus Toucan samples); per-sample sdg_model"
            ),
        ),
        Source(
            key="nemotron-v2-customer-service",
            dataset=NEMOTRON_V2,
            revision=NEMOTRON_V2_REV,
            files=("data/interactive_agent.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            generator_field="model",
            excluded_reason=(
                "each sample names its agent model, but neither the card nor the samples name "
                "the model that wrote the user turns and tool outputs"
            ),
        ),
        Source(
            key="nemotron-v2-search",
            dataset=NEMOTRON_V2,
            revision=NEMOTRON_V2_REV,
            files=("data/search.jsonl",),
            licence="CC-BY-4.0",
            kind="nemotron",
            excluded_reason="the card and the samples name no generator model",
        ),
        Source(
            key="openthoughts3",
            dataset=OPENTHOUGHTS,
            revision=OPENTHOUGHTS_REV,
            # The code block: the only allowed upstream source (OpenCodeReasoning) is there.
            files=tuple(f"data/train-{n:05d}-of-00120.parquet" for n in range(6)),
            licence="Apache-2.0",
            kind="openthoughts",
            card_generators=("QwQ-32B",),
            upstream_field="source",
            upstream_allow=OPENTHOUGHTS_UPSTREAM,
            note="card: answers by QwQ-32B; questions from the upstream sources in the paper",
        ),
    )
}

# Which mix bucket each source feeds.
BUCKETS = {
    "nemotron": ("nemotron-v1-tool-calling", "nemotron-v2-tool-calling"),
    "openthoughts": ("openthoughts3",),
}
