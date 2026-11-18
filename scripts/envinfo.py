"""Print environment information as JSON, for result provenance (DECISION.md section 6, rule 2).

Output keys: git_sha, git_dirty, utc_time, cpu_model, cpu_flags, cores_physical, cores_logical,
ram_bytes, os_name, os_version, machine, python_version, packages.

Deliberately omitted: hostname, username, home directory and the interpreter path, because
those identify the person. This file is copied per repo and has no cross-repo dependency.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent.parent

# Packages whose versions matter for reproducing a result. Missing packages report null.
PACKAGES: tuple[str, ...] = (
    "benchlint",
    "pydantic",
    "typer",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "datasketch",
    "datasets",
    "huggingface_hub",
)


def _run(cmd: list[str]) -> str | None:
    """Run a command and return its stripped stdout, or None if it fails or is missing."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _linux_cpuinfo_field(field: str) -> str | None:
    """Return the first value of a /proc/cpuinfo field such as 'model name' or 'flags'."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == field:
            return value.strip()
    return None


def git_sha() -> str | None:
    """Full commit SHA of this repo's HEAD, or None if there is no commit yet."""
    return _run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"])


# Run outputs live under results/ and are excluded from the dirty check: a run writes there, so
# counting them would make every result dirty. Code, specs, docs and scripts are what counts.
_EXCLUDED = (".", ":(exclude)results")


def _run_raw(cmd: list[str]) -> str | None:
    """Like _run, but keeps the output exactly (leading spaces in git status are meaningful)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_dirty_paths() -> list[str] | None:
    """Sorted paths with uncommitted changes outside results/, or None if git is unavailable."""
    out = _run_raw(["git", "-C", str(REPO_DIR), "status", "--porcelain", "--", *_EXCLUDED])
    if out is None:
        return None
    paths = {line[3:] for line in out.splitlines() if line.strip()}
    return sorted(paths)


def git_dirty() -> bool | None:
    """True if code, specs, scripts or docs have uncommitted changes (results/ excluded).

    None if git is unavailable. Before this change the check also counted results/, so a run
    that wrote its own outputs was recorded as dirty even on a clean tree.
    """
    paths = git_dirty_paths()
    if paths is None:
        return None
    return bool(paths)


def git_diff_sha256() -> str | None:
    """SHA-256 of the uncommitted code diff plus the untracked code files (results/ excluded).

    Two runs with the same value were made from the same code tree, so the value identifies the
    working tree a result came from. None if git is unavailable; a clean tree gives the hash of
    empty input.
    """
    diff = _run_raw(["git", "-C", str(REPO_DIR), "diff", "HEAD", "--", *_EXCLUDED])
    untracked = _run_raw(
        ["git", "-C", str(REPO_DIR), "ls-files", "--others", "--exclude-standard", "--", *_EXCLUDED]
    )
    if diff is None or untracked is None:
        return None
    payload = diff + "\n--untracked--\n" + "\n".join(sorted(untracked.splitlines())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cpu_model() -> str | None:
    """CPU brand string (macOS sysctl, or the Linux /proc/cpuinfo model name)."""
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return _linux_cpuinfo_field("model name") or platform.processor() or None


def cpu_flags() -> list[str]:
    """Sorted, upper-cased CPU feature flags, for example AVX2 and FMA."""
    if sys.platform == "darwin":
        parts: list[str] = []
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            value = _run(["sysctl", "-n", key])
            if value:
                parts.extend(value.split())
    else:
        value = _linux_cpuinfo_field("flags") or ""
        parts = value.split()
    return sorted({p.upper() for p in parts})


def _sysctl_int(key: str) -> int | None:
    value = _run(["sysctl", "-n", key])
    if value is None or not value.isdigit():
        return None
    return int(value)


def cores_physical() -> int | None:
    """Physical core count (macOS only; None on Linux where it is not read here)."""
    if sys.platform == "darwin":
        return _sysctl_int("hw.physicalcpu")
    return None


def cores_logical() -> int | None:
    """Logical CPU count as seen by the OS."""
    return os.cpu_count()


def ram_bytes() -> int | None:
    """Installed RAM in bytes."""
    if sys.platform == "darwin":
        return _sysctl_int("hw.memsize")
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * 1024
    return None


def os_version() -> str:
    """macOS product version (for example 13.6), or the kernel release elsewhere."""
    if sys.platform == "darwin":
        return platform.mac_ver()[0] or platform.release()
    return platform.release()


def package_versions() -> dict[str, str | None]:
    """Installed version of each package in PACKAGES, or None if not installed."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect() -> dict[str, Any]:
    """Gather all environment fields into one dict."""
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "git_dirty_paths": git_dirty_paths(),
        "git_diff_sha256": git_diff_sha256(),
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_model": cpu_model(),
        "cpu_flags": cpu_flags(),
        "cores_physical": cores_physical(),
        "cores_logical": cores_logical(),
        "ram_bytes": ram_bytes(),
        "os_name": platform.system(),
        "os_version": os_version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "packages": package_versions(),
    }


def main() -> int:
    """Print the environment JSON to stdout."""
    print(json.dumps(collect(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
