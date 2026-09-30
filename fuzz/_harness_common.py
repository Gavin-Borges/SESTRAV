"""Shared plumbing for the atheris harnesses in ``fuzz/``.

Imported by every ``fuzz_*.py`` harness after ``_offline_guard.install_and_verify()``
and before ``atheris.instrument_imports()``. Standard library only, so it is never
instrumented itself and adds no coverage noise.

Run modes
---------
strict (default)
    An exception outside the target's documented contract propagates, atheris
    reports it as a crash and libFuzzer writes the input under ``-artifact_prefix``.
    Use this mode in CI, to reproduce an input, and with ``-minimize_crash=1``.
record (``SESTRAV_FUZZ_RECORD_DIR`` set)
    The same exceptions are caught instead. The FIRST input for each distinct
    signature (exception type plus innermost first-party frame) is written to that
    directory with its traceback and the seconds elapsed since
    ``SESTRAV_FUZZ_RUN_T0`` (or process start), and fuzzing continues. One
    time-boxed run then reports every distinct failure instead of stopping at the
    first one.

``SESTRAV_FUZZ_SRC_ROOT`` points the harness at another checkout's ``src/``
package. It exists so a planted-bug copy of a target module can be fuzzed without
touching the working tree; the harness prints which root it imported from.

The ``fuzz_*.py`` files carry a ``#!/usr/bin/env python3`` line and are committed
executable because libFuzzer's ``-minimize_crash=1`` and ``-fork=N`` modes
re-execute ``argv[0]`` directly. Measured with atheris 3.0.0: a non-executable
script under ``-fork=1`` makes every job exit 127, which libFuzzer then counts as
a crash.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import traceback
from typing import Any, Callable, Iterable

CPU_LIMIT_SECONDS = 1.0
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_EITHER_CASE = frozenset(AA_ALPHABET + AA_ALPHABET.lower())

# Peptides the two parser harnesses may place in ``peptide_list``. Sources, all
# tracked: the literals in tests/test_external_predictors.py (first four) and the
# ``peptide`` column of
# data/allele_aware/IEDB-20260704-MULTI_VIRUS_ALLELE_AWARE-v2_holdouts.csv.
PEPTIDE_POOL = (
    "GLFYTRTGL",
    "AAYSDQWAL",
    "GLF",
    "AAY",
    "AVFDRKSDAK",
    "CLGGLLTMV",
    "FLRGRAYGL",
    "GLCTLVAML",
    "HPVGEADYFEY",
    "IVTDFSVIK",
    "KLPQLCTEL",
    "LLMGTLGIV",
    "RAHYNIVTF",
    "RAKFKQLL",
    "RPPIFIRRL",
    "TIHDIILECV",
)
MAX_PEPTIDES = 4
SYNTH_FLAG = 0x80
MAX_SYNTH_LEN = 15


class SlowInputError(Exception):
    """One input consumed more than CPU_LIMIT_SECONDS of process CPU time."""


class ContractViolation(AssertionError):
    """The target returned a value outside its documented contract."""


def repo_root() -> str:
    override = os.environ.get("SESTRAV_FUZZ_SRC_ROOT")
    if override:
        return os.path.abspath(override)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_import_path() -> str:
    """Put the chosen root first on sys.path so ``import src`` resolves there."""
    root = repo_root()
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)
    return root


def decode_text(data: bytes) -> str:
    """Decode fuzz bytes the way requests decodes a response body it cannot trust."""
    return data.decode("utf-8", "replace")


def split_parser_input(data: bytes) -> tuple[list[str], str]:
    """Split one fuzz input into ``(peptide_list, response_text)``.

    Byte 0: the low 7 bits modulo (MAX_PEPTIDES + 1) give the peptide count. When
    bit 7 is set and the count is non-zero, the first peptide is synthesised: the
    next byte gives its length (modulo MAX_SYNTH_LEN, plus 1) and that many bytes
    are mapped onto AA_ALPHABET. Each remaining peptide is one byte, an index into
    PEPTIDE_POOL. Everything after that is the response body, decoded as UTF-8.

    Peptides are always non-empty upper-case amino-acid strings, which is what
    ``get_predictor_dataframe`` hands the query functions after its own cleaning.
    """
    if not data:
        return [], ""
    head = data[0]
    count = (head & 0x7F) % (MAX_PEPTIDES + 1)
    pos = 1
    peptides: list[str] = []
    if head & SYNTH_FLAG and count:
        count -= 1
        if pos < len(data):
            length = data[pos] % MAX_SYNTH_LEN + 1
            pos += 1
            chunk = data[pos : pos + length]
            pos += len(chunk)
            synth = "".join(AA_ALPHABET[b % len(AA_ALPHABET)] for b in chunk)
            if synth:
                peptides.append(synth)
    for _ in range(count):
        if pos >= len(data):
            break
        peptides.append(PEPTIDE_POOL[data[pos] % len(PEPTIDE_POOL)])
        pos += 1
    return peptides, decode_text(data[pos:])


def encode_parser_input(pool_indices: Iterable[int], text: str) -> bytes:
    """Inverse of split_parser_input for pool-only peptide lists (seed building)."""
    indices = list(pool_indices)
    if len(indices) > MAX_PEPTIDES:
        raise ValueError(f"at most {MAX_PEPTIDES} peptides per input")
    if any(not 0 <= i < len(PEPTIDE_POOL) for i in indices):
        raise ValueError("pool index out of range")
    return bytes([len(indices)]) + bytes(indices) + text.encode("utf-8")


def call_with_cpu_limit(fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)`` and raise SlowInputError past CPU_LIMIT_SECONDS of CPU time.

    Process CPU time rather than wall time, so a contended host does not
    manufacture slow inputs. The limit is measured under atheris instrumentation,
    which slows interpreted code; re-time any hit uninstrumented before calling it
    a bug. The limit is applied whether ``fn`` returns or raises, so a slow input
    that ends in a documented exception is still reported.
    """
    start = time.process_time()
    try:
        return fn(*args)
    finally:
        spent = time.process_time() - start
        if spent > CPU_LIMIT_SECONDS:
            name = getattr(fn, "__name__", repr(fn))
            raise SlowInputError(f"{name} used {spent:.3f} s of CPU on one input")


class Runner:
    """Applies the strict or record behaviour described in the module docstring."""

    def __init__(self, target: str) -> None:
        # Silence the logging module for the whole harness process. The targets'
        # log lines are not part of any contract checked here, and
        # parse_netchop_html logs a warning for every body it cannot parse, which
        # is most fuzz inputs: the first 900 s run of that harness wrote 433 MB of
        # identical warnings to stderr. Harness output goes through sys.stderr.
        logging.disable(logging.CRITICAL)
        self.target = target
        self.root = repo_root()
        self.record_dir = os.environ.get("SESTRAV_FUZZ_RECORD_DIR") or None
        self.t0 = float(os.environ.get("SESTRAV_FUZZ_RUN_T0") or time.time())
        self.seen: set[str] = set()
        if self.record_dir:
            os.makedirs(self.record_dir, exist_ok=True)

    def announce(self, fn: Callable[..., Any]) -> None:
        source = getattr(getattr(fn, "__code__", None), "co_filename", "?")
        origin = "SESTRAV_FUZZ_SRC_ROOT" if os.environ.get("SESTRAV_FUZZ_SRC_ROOT") else "repo"
        sys.stderr.write(
            f"FUZZ_HARNESS target={self.target} module={self._rel(source)} "
            f"root={origin} mode={'record' if self.record_dir else 'strict'}\n"
        )

    def _rel(self, filename: str) -> str:
        path = os.path.abspath(filename)
        if path.startswith(self.root + os.sep):
            return os.path.relpath(path, self.root).replace(os.sep, "/")
        parts = path.replace(os.sep, "/").split("/")
        return "<ext>/" + "/".join(parts[-2:])

    def _scrub(self, text: str) -> str:
        text = text.replace(self.root + os.sep, "<root>/")
        for prefix in {sys.prefix, sys.base_prefix, os.path.expanduser("~")}:
            if prefix and len(prefix) > 1:
                text = text.replace(prefix, "<prefix>")
        return text

    def signature(self, exc: BaseException) -> str:
        frames = traceback.extract_tb(exc.__traceback__)
        chosen = None
        for frame in reversed(frames):
            if self._rel(frame.filename).startswith("src/"):
                chosen = frame
                break
        if chosen is None and frames:
            chosen = frames[-1]
        where = f"{self._rel(chosen.filename)}#{chosen.lineno}" if chosen else "no-frame"
        return f"{type(exc).__name__}@{where}"

    def run(self, body: Callable[[bytes], None], data: bytes) -> None:
        if self.record_dir is None:
            body(data)
            return
        try:
            body(data)
        except Exception as exc:
            self._record(exc, data)

    def _record(self, exc: Exception, data: bytes) -> None:
        sig = self.signature(exc)
        if sig in self.seen:
            return
        self.seen.add(sig)
        record_dir = self.record_dir or ""
        base = os.path.join(record_dir, hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16])
        if os.path.exists(base + ".txt"):
            return
        elapsed = time.time() - self.t0
        with open(base + ".input", "wb") as fh:
            fh.write(data)
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with open(base + ".txt", "w", encoding="utf-8") as fh:
            fh.write(f"target={self.target}\nsignature={sig}\n")
            fh.write(f"elapsed_seconds={elapsed:.3f}\ninput_bytes={len(data)}\n")
            fh.write(self._scrub(detail))
        sys.stderr.write(f"FUZZ_RECORD new signature {sig} at {elapsed:.3f} s\n")
