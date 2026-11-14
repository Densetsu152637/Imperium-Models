"""Processes the pipeline starts, and only those: tracked by PID and stopped by PID."""

import os
import subprocess
import time
from pathlib import Path


def find_llama_tool(
    tool: str, explicit: str | Path | None = None, env: str | None = None
) -> Path | None:
    """A llama.cpp binary: option, environment variable, PATH, then Imperium's runtime."""
    import shutil

    for candidate in (explicit, os.environ.get(env) if env else None):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    on_path = shutil.which(tool)
    if on_path:
        return Path(on_path)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        runtimes = Path(local) / "Imperium" / "data" / "runtimes" / "llama.cpp"
        name = f"{tool}.exe" if os.name == "nt" else tool
        found = sorted(runtimes.glob(f"*/*/bin/{name}"), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    return None


def find_llama_server(explicit: str | Path | None = None) -> Path | None:
    """The llama-server binary: option, EMPEROR_LLAMA_SERVER, PATH, then Imperium's runtime."""
    return find_llama_tool("llama-server", explicit, "EMPEROR_LLAMA_SERVER")


def process_parents() -> dict[int, int]:
    """Every running process's parent, by PID; {} where this cannot be read."""
    if os.name == "nt":
        return _windows_parents()
    parents = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            parents[int(stat.parent.name)] = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
    return parents


def _windows_parents() -> dict[int, int]:
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        return {}
    parents: dict[int, int] = {}
    entry = Entry()
    entry.dwSize = ctypes.sizeof(Entry)
    try:
        ok = kernel.Process32First(snapshot, ctypes.byref(entry))
        while ok:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            ok = kernel.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return parents


def process_created(pid: int) -> int | None:
    """A process's creation time (comparable between processes on this machine), or None
    when the process cannot be opened or has exited."""
    if os.name == "nt":
        return _windows_created(pid)
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])  # starttime, in clock ticks since boot
    except (OSError, IndexError, ValueError):
        return None


def _windows_created(pid: int) -> int | None:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        created = times[0]
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def with_descendants(roots, parents: dict[int, int], created=process_created) -> tuple[int, ...]:
    """`roots` and every process descended from them.

    Windows keeps a dead parent's PID in its children's records, so a reused PID could adopt a
    foreign process tree. An edge counts only when both processes can be opened and the child
    was created no earlier than the parent (`created` gives creation times)."""
    found = set(roots)
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        if pid != parent:
            children.setdefault(parent, []).append(pid)
    times: dict[int, int | None] = {}

    def time_of(pid: int) -> int | None:
        if pid not in times:
            times[pid] = created(pid)
        return times[pid]

    def real_edge(parent: int, child: int) -> bool:
        parent_time, child_time = time_of(parent), time_of(child)
        return parent_time is not None and child_time is not None and child_time >= parent_time

    stack = list(found)
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child not in found and real_edge(parent, child):
                found.add(child)
                stack.append(child)
    return tuple(sorted(found))


class OwnProcesses:
    """Every process the pipeline launched, and their descendants; the idle gate subtracts
    these from GPU load. Descendants matter on Windows, where a virtual environment's
    python.exe is a launcher that runs the real interpreter as its child."""

    def add(self, proc: subprocess.Popen) -> None:
        self._procs[proc.pid] = proc

    def __init__(self, parents=process_parents, created=process_created) -> None:
        self._procs: dict[int, subprocess.Popen] = {}
        self._parents = parents
        self._created = created

    def pids(self) -> tuple[int, ...]:
        alive = {pid: p for pid, p in self._procs.items() if p.poll() is None}
        self._procs = alive
        if not alive:
            return ()
        return with_descendants(alive, self._parents(), self._created)

    def terminate_all(self, grace: float = 20.0) -> None:
        """Stop every still-running child we started, by PID, and nothing else."""
        for proc in list(self._procs.values()):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(grace)
                except subprocess.TimeoutExpired:
                    proc.kill()


class ManagedProcess:
    """A child process with a log file; `stop` terminates it by PID (and nothing else)."""

    def __init__(self, command: list[str], log_path: Path, own: OwnProcesses, env=None):
        self.command = command
        self.log_path = log_path
        self.own = own
        self.env = env
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> "ManagedProcess":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            self.command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self.env,
            creationflags=flags,
        )
        self.own.add(self.proc)
        return self

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def poll(self) -> int | None:
        return self.proc.poll() if self.proc else None

    def wait(self, timeout: float) -> int | None:
        try:
            return self.proc.wait(timeout) if self.proc else None
        except subprocess.TimeoutExpired:
            return None

    def stop(self, grace: float = 20.0) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(grace)
        if self._log:
            self._log.close()
            self._log = None


def llama_server_command(
    binary: Path,
    model: Path,
    port: int,
    context: int,
    parallel: int = 1,
    vram_margin_mib: int = 1536,
) -> list[str]:
    """llama-server on loopback with the model's own chat template (tool calls).

    `--fit` puts as many layers in VRAM as fit with `vram_margin_mib` to spare and keeps the rest
    on the CPU, so the model runs partly on the GPU beside whatever else is using it.
    """
    return [
        str(binary),
        "-m", str(model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(context * parallel),
        "-np", str(parallel),
        "-ngl", "auto",
        "--fit", "on",
        "--fit-target", str(vram_margin_mib),
        "--jinja",
    ]  # fmt: skip


def wait_for_health(
    url_root: str, process: ManagedProcess, seconds: float = 600.0, interval: float = 2.0
) -> bool:
    """Poll `<url_root>/health` until it answers 200; False if the process dies or time runs out."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url_root}/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    return False
