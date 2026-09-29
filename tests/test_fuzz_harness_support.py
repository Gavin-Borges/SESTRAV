"""Offline checks for the atheris harness support code under ``fuzz/``.

The harnesses themselves need atheris, which is not part of the ``dev`` extra, so
this module never imports a harness. It checks the three properties that make a
harness run meaningful and that can be verified without atheris:

1. The offline guard refuses every network entry point it patches, and its
   positive control really can fail (a control that cannot fail proves nothing).
   Both run in a subprocess so the guard never patches the pytest process.
2. Every harness installs the guard BEFORE it imports atheris and before
   ``atheris.instrument_imports()`` runs, read from the source with ast.
3. The seed-corpus builder produces a non-empty corpus for every harness from
   tracked files, and the parser-input encoding it uses round-trips through the
   decoding the harnesses use.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[1]
FUZZ_DIR = REPO_ROOT / "fuzz"
HARNESSES = sorted(FUZZ_DIR.glob("fuzz_*.py"))


def _load_harness_common() -> ModuleType:
    # Loaded from its path rather than imported: fuzz/ is not a package and is not on
    # sys.path, and tests/test_dev_extra_runs_the_test_suite.py reads every
    # module-scope import statement in tests/ as a third-party dependency.
    spec = importlib.util.spec_from_file_location(
        "fuzz_harness_common", FUZZ_DIR / "_harness_common.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_harness_common()

GUARD_OK = """
import sys
sys.path.insert(0, sys.argv[1])
import _offline_guard
for line in _offline_guard.install_and_verify(quiet=True):
    print(line)
"""

# Install the guard, then undo ONE patch the way a broken guard would. getaddrinfo
# is replaced by a stub that returns no addresses, so nothing leaves the host even
# though the guard is no longer refusing it; the control must notice and abort.
GUARD_BROKEN = """
import socket, sys
sys.path.insert(0, sys.argv[1])
import _offline_guard
_offline_guard.install()
socket.getaddrinfo = lambda *args, **kwargs: []
_offline_guard.positive_control()
print("control did not abort")
"""


# Starts from the worst inherited state: a caller's bypass list naming the host, and
# the no_proxy="*" an earlier version of the guard itself set.
GUARD_PROXY = """
import os, sys, urllib.request
os.environ["NO_PROXY"] = "example.org"
os.environ["no_proxy"] = "*"
sys.path.insert(0, sys.argv[1])
import _offline_guard
_offline_guard.install()
print("urllib bypass", urllib.request.proxy_bypass("example.org"))
print("urllib https", urllib.request.getproxies().get("https"))
try:
    import requests.utils
except ImportError:
    print("requests absent")
else:
    print("requests https", requests.utils.get_environ_proxies("https://example.org/").get("https"))
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, str(FUZZ_DIR)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )


def test_offline_guard_positive_control_passes_when_installed():
    proc = _run(GUARD_OK)
    assert proc.returncode == 0, proc.stderr
    assert "OFFLINE_GUARD control verdict: PASS" in proc.stdout
    refused = [ln for ln in proc.stdout.splitlines() if ln.endswith("(OfflineGuardViolation)")]
    assert len(refused) >= 4, proc.stdout


def test_offline_guard_proxy_layer_exempts_no_host():
    """The proxy variables are the only layer a child process inherits.

    A bypass list defeats them: no_proxy="*" exempts every host, so the dead proxy
    was never used. After install() no host may be exempted, including one named by
    a bypass list inherited from the caller.
    """
    proc = _run(GUARD_PROXY)
    assert proc.returncode == 0, proc.stderr
    assert "urllib bypass False" in proc.stdout, proc.stdout
    assert "urllib https http://127.0.0.1:1" in proc.stdout, proc.stdout
    assert "requests absent" in proc.stdout or "requests https http://127.0.0.1:1" in proc.stdout, (
        proc.stdout
    )


def test_offline_guard_positive_control_aborts_when_a_patch_is_missing():
    proc = _run(GUARD_BROKEN)
    assert proc.returncode == 97, (proc.returncode, proc.stdout, proc.stderr)
    assert "control did not abort" not in proc.stdout
    assert "socket.getaddrinfo: FAIL did not raise" in proc.stderr


def _first_line(tree: ast.Module, predicate) -> int | None:
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Call, ast.Import)) and predicate(node)
    ]
    return min(lines) if lines else None


def _is_guard_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "install_and_verify"
    )


def _is_atheris_import(node: ast.AST) -> bool:
    return isinstance(node, ast.Import) and any(a.name == "atheris" for a in node.names)


def _is_instrument_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "instrument_imports"
    )


def test_there_are_harnesses_to_check():
    assert len(HARNESSES) >= 4, [p.name for p in HARNESSES]


@pytest.mark.parametrize("harness", HARNESSES, ids=lambda p: p.stem)
def test_harness_installs_offline_guard_before_atheris(harness: Path):
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    guard = _first_line(tree, _is_guard_call)
    atheris_import = _first_line(tree, _is_atheris_import)
    instrument = _first_line(tree, _is_instrument_call)
    assert guard is not None, "offline guard is never installed"
    assert atheris_import is not None, "harness does not import atheris"
    assert instrument is not None, "harness does not call atheris.instrument_imports"
    assert guard < atheris_import < instrument, (guard, atheris_import, instrument)


def test_seed_builder_covers_every_harness(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, str(FUZZ_DIR / "build_seed_corpora.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    built = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert built == {p.stem for p in HARNESSES}
    for name in built:
        assert any((tmp_path / name).iterdir()), f"empty seed corpus for {name}"


@given(
    st.lists(
        st.integers(min_value=0, max_value=len(common.PEPTIDE_POOL) - 1),
        max_size=common.MAX_PEPTIDES,
    ),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
)
def test_parser_input_encoding_round_trips(indices: list[int], text: str):
    encoded = common.encode_parser_input(indices, text)
    peptides, decoded = common.split_parser_input(encoded)
    assert peptides == [common.PEPTIDE_POOL[i] for i in indices]
    assert decoded == text


@given(st.binary(max_size=64))
def test_split_parser_input_always_yields_nonempty_amino_acid_peptides(data: bytes):
    peptides, _text = common.split_parser_input(data)
    assert len(peptides) <= common.MAX_PEPTIDES
    for pep in peptides:
        assert pep and set(pep) <= set(common.AA_ALPHABET)
