"""SEC-3 / CodeQL alert #79 (py/log-injection, CWE-117): api/main.py log-forging gate.

api/main.py is the project's only HTTP entrypoint, and its /score handler put the
caller's peptide, allele and the raised exception straight into a single-line log
record:

    logger.error(f"Scoring error for {body.sequence}/{body.allele}: {exc}")

The configured format is one record per line ("%(asctime)s [%(levelname)s]
%(message)s"), so a CR or LF inside any of those three closes the record and lets
the caller write a second, fully-formed one: a forged ERROR in the audit trail, or
a real one pushed out of a viewer's window.

Reachability, stated plainly so this file is not misread as an exploit report: the
payloads below CANNOT be delivered over HTTP today. `sequence` and `allele` carry
anchored Pydantic patterns, and pydantic 2.13.4 compiles them with the Rust regex
engine, which rejects every CR/LF variant tested against them. These tests
therefore build the request body with `PeptideInput.model_construct`, which skips
validation. That is the point, not a shortcut: the sanitizer exists for the day the
patterns are widened (HLA-C, three-field alleles like 02:01:01) or the engine
changes underneath them. The identical anchored pattern under Python's own
`re.match` ALREADY accepts a trailing newline, where `re.fullmatch` does not.
`model_construct` simulates that future and pins the sink's behaviour under it.

The first two tests assert on the record api/main.py's own logging path actually
emits, not on the helper in isolation, so swapping the sanitized call site back to
an f-string fails them. The AST guard extends that to every logging call in the
module, because a bypass can be introduced at a call site no test exercises.
"""

from __future__ import annotations

import ast
import logging
import pathlib
from collections.abc import Iterator

import pytest
from fastapi import HTTPException

import api.main as api_main

# Mirrors the format string api/main.py hands to logging.basicConfig. Rendering
# through it is the whole point: log forging is only meaningful against the real
# one-record-per-line layout.
APP_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# A payload that reads as a valid peptide up to the CRLF and then continues with a
# complete, plausible line in the app's own format. Without the sanitizer this lands
# in the log as a second ERROR record that no code ever emitted.
FORGED_LINE = "2026-08-22 03:14:15 [ERROR] auth: admin session opened for peptide review"
HOSTILE_SEQUENCE = f"GILGFVFTL\r\n{FORGED_LINE}"
HOSTILE_ALLELE = "HLA-A*02:01\nHLA-B*07:02"

MODULE_PATH = pathlib.Path(api_main.__file__)
SANITIZER_NAME = "_sanitize_for_log"

# Same set test_encoding_ascii_output.py treats as emitting calls.
LOGGING_METHODS = {"info", "warning", "error", "debug", "critical", "exception"}

# Substrings marking an interpolated expression as carrying request-derived text.
# `exc` is included because an exception raised from _score_peptide is built from
# the caller's own peptide and allele - see PeptideNotInPanelError(sequence).
REQUEST_DERIVED_MARKERS = ("body.", "sequence", "allele", "exc")


class _CapturingHandler(logging.Handler):
    """Captures records both raw and fully rendered, as a log file would receive them."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(APP_LOG_FORMAT))
        self.records: list[logging.LogRecord] = []
        self.rendered: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.rendered.append(self.format(record))


@pytest.fixture
def captured_log() -> Iterator[_CapturingHandler]:
    handler = _CapturingHandler()
    previous_level = api_main.logger.level
    api_main.logger.setLevel(logging.INFO)
    api_main.logger.addHandler(handler)
    try:
        yield handler
    finally:
        api_main.logger.removeHandler(handler)
        api_main.logger.setLevel(previous_level)


@pytest.fixture
def exploding_scorer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drives /score into its generic `except Exception` branch - the alert #79 sink.

    The stand-in quotes the caller's peptide back in its exception message. That is
    not contrived: PeptideNotInPanelError(sequence) in the same module does exactly
    this, which is why `exc` is sanitized alongside the two named fields rather than
    being assumed to be developer-controlled text.
    """

    def _boom(sequence: str, allele: str) -> tuple[float, float | None]:
        raise RuntimeError(f"feature extraction failed for {sequence}")

    monkeypatch.setattr(api_main, "_score_peptide", _boom)
    api_main._manager._loaded = True
    try:
        yield
    finally:
        api_main._manager._loaded = False


def _post_score(sequence: str, allele: str) -> HTTPException:
    """Calls the real handler with validation skipped; returns the 500 it raises."""
    body = api_main.PeptideInput.model_construct(sequence=sequence, allele=allele)
    with pytest.raises(HTTPException) as excinfo:
        api_main.score_peptide(body)
    assert excinfo.value.status_code == 500
    return excinfo.value


def test_crlf_payload_cannot_forge_a_second_log_line(
    captured_log: _CapturingHandler,
    exploding_scorer: None,
) -> None:
    """The invariant: one request produces exactly one log line, whatever it contains."""
    _post_score(HOSTILE_SEQUENCE, HOSTILE_ALLELE)

    assert len(captured_log.rendered) == 1, (
        f"expected one record, got {len(captured_log.rendered)}: {captured_log.rendered}"
    )
    rendered = captured_log.rendered[0]

    assert "\r" not in rendered, f"CR survived into the emitted record: {rendered!r}"
    assert "\n" not in rendered, f"LF survived into the emitted record: {rendered!r}"
    assert len(rendered.splitlines()) == 1, (
        "the emitted log record spans multiple lines, so a caller can forge one:\n" + rendered
    )
    assert "\r" not in captured_log.records[0].getMessage()
    assert "\n" not in captured_log.records[0].getMessage()

    # Neutralised, not discarded. An operator must still see what was attempted -
    # a sanitizer that silently drops the field destroys the evidence instead of
    # the attack, and would pass the assertions above for the wrong reason.
    assert "auth: admin session opened" in rendered
    assert "HLA-B*07:02" in rendered
    assert "GILGFVFTL" in rendered


def test_oversized_field_cannot_flood_a_log_record(
    captured_log: _CapturingHandler,
    exploding_scorer: None,
) -> None:
    """The second, independent control: one field is bounded, so it cannot roll the log."""
    _post_score("A" * 10_000, "HLA-A*02:01")

    rendered = captured_log.rendered[0]
    assert "truncated" in rendered, "an oversized field was logged in full"
    assert len(rendered) < 1_000, f"one record grew to {len(rendered)} chars"


@pytest.mark.parametrize(
    "payload",
    [
        "GILGFVFTL\r\nforged",
        "GILGFVFTL\nforged",
        # The bare CR is the case CodeQL's published "good" example misses; it still
        # returns the cursor to column zero and overwrites the line in many viewers.
        "GILGFVFTL\rforged",
        "GILGFVFTL\n\rforged",
    ],
)
def test_sanitizer_strips_every_line_separator_form(payload: str) -> None:
    out = api_main._sanitize_for_log(payload)
    assert "\r" not in out
    assert "\n" not in out
    assert len(out.splitlines()) == 1
    assert out == "GILGFVFTLforged"


def _logging_calls(tree: ast.Module) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LOGGING_METHODS
        ):
            yield node


def _interpolated_values(call: ast.Call) -> list[ast.expr]:
    """Expressions whose VALUE is rendered into the record: lazy args and f-string parts."""
    values: list[ast.expr] = list(call.args[1:])
    values += [node.value for node in ast.walk(call) if isinstance(node, ast.FormattedValue)]
    return values


def _is_sanitizer_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SANITIZER_NAME
    )


def test_every_request_derived_logging_value_goes_through_the_sanitizer() -> None:
    """Extends the runtime gate above to call sites no test exercises.

    The runtime tests cover the /score handler. api/main.py logs from other places
    too, and the MHCflurry fallback in _score_peptide interpolates an exception
    raised from the caller's own peptide and allele. A source-level check is the
    only thing that keeps a newly added, unsanitized call site from shipping.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    violations = []
    checked = 0
    for call in _logging_calls(tree):
        for value in _interpolated_values(call):
            text = ast.unparse(value)
            if not any(marker in text for marker in REQUEST_DERIVED_MARKERS):
                continue
            checked += 1
            if _is_sanitizer_call(value):
                continue
            violations.append(f"api/main.py:{value.lineno} logs `{text}` unsanitized")

    assert checked, (
        "this guard matched no request-derived logging value at all, so it has gone "
        f"vacuous - check that {REQUEST_DERIVED_MARKERS} still describes api/main.py"
    )
    assert not violations, (
        "request-derived value(s) reach a log record without "
        f"{SANITIZER_NAME}(), re-opening CWE-117 log forging "
        "(CodeQL py/log-injection alert #79):\n" + "\n".join(violations)
    )
