"""Fresh-namespace checks for Python chunks in docs/results_report.qmd."""

import ast
import builtins
from pathlib import Path
import re
import sys
import types


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
    loaded = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    defined = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
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
