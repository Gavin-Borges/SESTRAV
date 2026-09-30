"""Fresh-namespace checks for Python chunks in docs/results_report.qmd."""

import ast
import builtins
from pathlib import Path
import re
import sys
import types

import pytest


REPORT = Path(__file__).parents[1] / "docs" / "results_report.qmd"

# A Quarto python chunk opener: ```{python}, optionally carrying chunk options
# inside the braces and trailing whitespace after them.
PYTHON_FENCE = re.compile(r"^```\{python\b[^}]*\}\s*$")


def _python_chunks() -> list[str]:
    chunks: list[str] = []
    current: list[str] | None = None
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        if PYTHON_FENCE.match(line):
            current = []
        elif line.rstrip() == "```" and current is not None:
            chunks.append("\n".join(current) + "\n")
            current = None
        elif current is not None:
            current.append(line)
    return chunks


def _python_fence_count() -> int:
    """Count report lines containing '```{python' anywhere, without PYTHON_FENCE.

    Deliberately looser than the extractor: an indented, four-backtick,
    unterminated or otherwise unrecognised opener is counted here but not
    extracted, so a chunk the extractor skips makes the two numbers disagree.
    """
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    return sum("```{python" in line for line in lines)


def _checked_chunks() -> list[str]:
    chunks = _python_chunks()
    fences = _python_fence_count()
    assert fences > 0, "report must contain at least one Python chunk"
    assert len(chunks) == fences, (
        f"extracted {len(chunks)} Python chunk(s) but counted {fences} "
        "'```{python' line(s); the extractor skipped a chunk"
    )
    return chunks


def _borrowed_names(source: str) -> set[str]:
    """Return the names a chunk loads but never binds, imports or gets from builtins.

    Bound forms: assignment and loop targets, imports, def and class names,
    function and lambda parameters, except-as names, and match-case captures
    (``case [a]``, ``case [*rest]``, ``case {**rest}``). A star import raises
    ValueError, because the names it binds cannot be known from the source.

    Known limit: the analysis is scope-insensitive. A name bound anywhere in a
    chunk counts as bound everywhere in it, so a parameter ``x`` of one function
    hides a module-level use of a borrowed ``x``, and a use that comes before
    its own assignment is not flagged. The exec test is the second instrument
    for that, but it covers only the branches each chunk actually takes.
    """
    tree = ast.parse(source)

    # A bare annotation binds NOTHING at runtime: `total: int` leaves `total`
    # undefined, so `total: int` followed by `print(total)` raises NameError.
    # Its target still carries Store ctx, so counting it as bound would hide a
    # borrowed name. Matched by node identity, not by name, so a name that is
    # annotated here and genuinely assigned elsewhere still counts as bound.
    annotation_only = {
        id(node.target)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and node.value is None
        and isinstance(node.target, ast.Name)
    }

    # An augmented assignment does not ESTABLISH its target either: it requires
    # the name to exist already. Its target node is therefore excluded from the
    # bound set below and added to the loaded set instead. Both are matched by
    # node identity, so `total = 0` followed by `total += 1` still counts the
    # plain assignment as the binding.
    augmented = {
        id(node.target)
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
    }

    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    defined = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and id(node) not in annotation_only
        and id(node) not in augmented
    }

    # An augmented assignment READS its target before writing it: `total += 1`
    # raises NameError when `total` is unbound. The target carries Store ctx, so
    # without both halves of this it counted as bound and never as loaded, which
    # hid a borrowed name behind the very statement that borrows it.
    loaded.update(
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            defined.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                raise ValueError(f"star import from {node.module!r} is unsupported")
            defined.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            defined.add(node.rest)
    return loaded - defined - set(dir(builtins))


def test_every_python_chunk_declares_its_loaded_names():
    chunks = _checked_chunks()
    borrowed = {index: sorted(_borrowed_names(source)) for index, source in enumerate(chunks, 1)}
    borrowed = {index: names for index, names in borrowed.items() if names}
    assert borrowed == {}, f"cross-chunk dependencies: {borrowed}"


def test_every_python_chunk_executes_in_a_fresh_namespace(monkeypatch, tmp_path):
    chunks = _checked_chunks()
    monkeypatch.chdir(tmp_path)
    display_module = types.ModuleType("IPython.display")
    display_module.display = lambda value: value
    display_module.Markdown = lambda value: value
    ipython_module = types.ModuleType("IPython")
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    for index, source in enumerate(chunks, 1):
        namespace = {"__name__": f"results_report_chunk_{index}"}
        # The source is the tracked report named by REPORT, not user input.
        exec(  # noqa: S102
            compile(source, f"results_report.qmd chunk {index}", "exec"), namespace
        )


# ---------------------------------------------------------------------------
# _borrowed_names: two forms whose target carries Store ctx but binds nothing
# ---------------------------------------------------------------------------
# Both were counted as BOUND, so a chunk borrowing a name through either one
# reported no cross-chunk dependency at all. Both raise NameError at runtime on
# an unbound name, which is the property this analysis exists to predict:
#
#     exec("total += 1", {})        -> NameError: name 'total' is not defined
#     exec("total: int\nprint(total)", {}) -> NameError: name 'total' is not defined
#
# The last four cases are false-positive controls: each binds the name for real,
# so widening the rule must not start reporting them.


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("print(total)\n", ["total"], id="plain_load"),
        pytest.param("total += 1\n", ["total"], id="augmented_assign"),
        pytest.param("total: int\nprint(total)\n", ["total"], id="bare_annotation"),
        pytest.param("total: int = 1\nprint(total)\n", [], id="annotation_with_value"),
        pytest.param("total: int\ntotal = 1\nprint(total)\n", [], id="annotated_and_assigned"),
        pytest.param("total = 0\ntotal += 1\n", [], id="augmented_after_local_bind"),
        pytest.param("for i in range(3):\n    i += 1\n", [], id="augmented_loop_variable"),
    ],
)
def test_borrowed_names_counts_a_name_as_bound_only_when_it_is(source, expected):
    """A Store ctx is not the same thing as a binding.

    `total += 1` READS its target before writing it, and a bare `total: int`
    records an annotation without creating the variable. Both targets carry
    Store ctx, so treating Store as "bound" hid a borrowed name behind the very
    statement that borrows it.
    """
    assert sorted(_borrowed_names(source)) == expected
