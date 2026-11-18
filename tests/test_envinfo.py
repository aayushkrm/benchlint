"""Tests for scripts/envinfo.py: it runs, prints the expected keys, and omits identifying data."""

from __future__ import annotations

import getpass
import json
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "envinfo.py"

EXPECTED_KEYS = {
    "git_sha",
    "git_dirty",
    "git_dirty_paths",
    "git_diff_sha256",
    "utc_time",
    "cpu_model",
    "cpu_flags",
    "cores_physical",
    "cores_logical",
    "ram_bytes",
    "os_name",
    "os_version",
    "machine",
    "python_version",
    "packages",
}


@pytest.fixture(scope="module")
def envinfo_stdout() -> str:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture(scope="module")
def envinfo(envinfo_stdout: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(envinfo_stdout)
    return data


def test_script_prints_expected_keys(envinfo: dict[str, Any]) -> None:
    assert EXPECTED_KEYS <= set(envinfo)


def test_utc_time_is_iso8601_utc(envinfo: dict[str, Any]) -> None:
    parsed = datetime.fromisoformat(envinfo["utc_time"])
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_git_sha_is_full_hash_or_null(envinfo: dict[str, Any]) -> None:
    sha = envinfo["git_sha"]
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_cpu_flags_are_sorted_uppercase_strings(envinfo: dict[str, Any]) -> None:
    flags = envinfo["cpu_flags"]
    assert isinstance(flags, list)
    assert flags == sorted(flags)
    assert all(isinstance(f, str) and f == f.upper() for f in flags)


def test_core_and_ram_fields_are_positive_ints_when_present(envinfo: dict[str, Any]) -> None:
    assert isinstance(envinfo["cores_logical"], int) and envinfo["cores_logical"] >= 1
    for key in ("cores_physical", "ram_bytes"):
        value = envinfo[key]
        assert value is None or (isinstance(value, int) and value > 0)


def test_packages_map_has_every_key(envinfo: dict[str, Any]) -> None:
    packages = envinfo["packages"]
    assert isinstance(packages, dict)
    assert "numpy" in packages and "pydantic" in packages
    assert all(v is None or isinstance(v, str) for v in packages.values())


def test_output_omits_hostname_username_and_paths(envinfo_stdout: str) -> None:
    lowered_keys = {k.lower() for k in json.loads(envinfo_stdout)}
    assert not any("host" in k or "user" in k or "home" in k for k in lowered_keys)
    assert str(Path.home()) not in envinfo_stdout
    assert sys.executable not in envinfo_stdout
    for secret in (socket.gethostname(), getpass.getuser()):
        if len(secret) >= 4:
            assert secret not in envinfo_stdout


def test_git_provenance_fields_have_the_expected_types(envinfo: dict[str, Any]) -> None:
    assert envinfo["git_dirty"] in (True, False, None)
    paths = envinfo["git_dirty_paths"]
    assert paths is None or (isinstance(paths, list) and all(isinstance(p, str) for p in paths))
    digest = envinfo["git_diff_sha256"]
    assert digest is None or re.fullmatch(r"[0-9a-f]{64}", digest)


def test_results_are_never_listed_as_dirty(envinfo: dict[str, Any]) -> None:
    """Run outputs under results/ do not make a tree dirty (see scripts/envinfo.py)."""
    for path in envinfo["git_dirty_paths"] or []:
        assert not path.startswith("results/"), path


def test_dirty_flag_matches_dirty_paths(envinfo: dict[str, Any]) -> None:
    if envinfo["git_dirty_paths"] is None:
        pytest.skip("git is unavailable")
    assert envinfo["git_dirty"] == bool(envinfo["git_dirty_paths"])
