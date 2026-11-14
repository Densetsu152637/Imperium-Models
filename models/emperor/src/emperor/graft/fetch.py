"""Sampling records from large Hugging Face dataset files without downloading them whole.

JSONL files are read in slices at evenly spread random byte offsets (HTTP range requests): each
slice skips its first, partial line and keeps the whole lines after it. Parquet shards are
downloaded whole (they are a few hundred MB) and read with pyarrow (`--group graft`).
Downloads go to a directory the caller names; nothing here runs downloaded code.
"""

import json
import random
import urllib.request
from collections.abc import Iterator
from pathlib import Path

RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/main/{file}"
USER_AGENT = "imperium-models-graft/1"


def file_url(dataset: str, file: str) -> str:
    return RESOLVE.format(dataset=dataset, file=file)


def _request(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=120)


def remote_size(url: str) -> int:
    with _request(url, {"Range": "bytes=0-0"}) as response:
        span = response.headers.get("Content-Range", "")
        if "/" in span:
            return int(span.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])


def read_range(url: str, start: int, end: int) -> bytes:
    with _request(url, {"Range": f"bytes={start}-{end}"}) as response:
        return response.read()


def whole_lines(chunk: bytes, at_start: bool) -> list[str]:
    """The complete lines in a slice: the first is dropped unless the slice starts the file,
    the last unless it ends with a newline."""
    lines = chunk.split(b"\n")
    if not at_start:
        lines = lines[1:]
    lines = lines[:-1]  # either empty (slice ended on a newline) or partial
    return [line.decode("utf-8") for line in lines if line.strip()]


def sample_jsonl(
    url: str,
    slices: int,
    slice_bytes: int,
    seed: int,
    *,
    size: int | None = None,
    reader=read_range,
) -> Iterator[dict]:
    """Records from `slices` random slices of a remote JSONL file, in random slice order."""
    size = size if size is not None else remote_size(url)
    rng = random.Random(seed)
    span = max(size - slice_bytes, 1)
    starts = sorted({rng.randrange(span) for _ in range(slices)})
    rng.shuffle(starts)  # a caller that stops early still gets the whole file's spread
    for start in starts:
        chunk = reader(url, start, min(start + slice_bytes, size) - 1)
        for line in whole_lines(chunk, at_start=start == 0):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a line longer than the slice is cut off: skipped


def download(url: str, out: Path) -> Path:
    """Fetch a file once (kept for later runs), writing under a temporary name first."""
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".part")
    with _request(url) as response, partial.open("wb") as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    partial.replace(out)
    return out


def parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq  # the `graft` dependency group

    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()
