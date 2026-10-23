"""The shipped config must resolve pipeline.smk's path defaults to TRACKED files.

`pipeline.smk` reads its inputs through `config.get("<key>", "<fallback path>")`. A
fallback that names a file which is not in the repository is invisible to CI, because
`.github/workflows/ci.yml` dry-runs with `--configfile tests/fixtures/dag_smoke/config.smoke.yaml`
and that fixture overrides every such key. The SHIPPED `config.yaml` has therefore never
been DAG-resolved by any automated check.

That gap shipped a real defect. `TRAINING_DATASET` fell back to
`data/immunogenicity_dataset_v4.csv`, which is gitignored, so every documented
`snakemake --snakefile pipeline.smk` command in README.md, USAGE.md and CONTRIBUTING.md
died with `MissingInputException in rule qc_dataset` on a fresh clone - while passing on
any workstation that still had the v4 file on disk. Measured both ways before this test
was written: exit 1 against a tracked-only export of `origin/main`, exit 0 in the working
checkout. A local green run was not evidence.

Anti-vacuity. This test fails if:
  - the `training_dataset` key is removed from config.yaml (the effective value falls back
    to the untracked v4 path);
  - `training_dataset` is repointed at any path that is not tracked;
  - a NEW `config.get("<key>", "<path>")` is added to pipeline.smk whose resolved value is
    untracked and which is not deliberately listed in KNOWN_UNTRACKED_DEFAULTS below.
It does NOT pass merely because the file exists on the developer's disk: tracking is
checked with `git ls-files`, not with `Path.exists()`. That distinction is the whole point.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "pipeline.smk"
CONFIG = REPO_ROOT / "config.yaml"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_human_proteome.py"

# Keys whose default is deliberately allowed to be untracked, each with the reason it is
# exempt. An exemption is recorded here so it is VISIBLE; the failure mode this whole file
# exists to catch is an untracked default that nobody knew about.
#
# reference_proteome: `pipeline.smk` falls back to
# data/proteomes/human_uniprot_UP000005640.fasta, which is not in the repository. It is
# exempt because it is NOT reached by the default target: with training_dataset set
# correctly, `snakemake --dry-run --cores 1` resolves to exit 0 on a tracked-only tree,
# measured. A large reference proteome is a download, not a repository artifact. If a
# future rule pulls it into the default DAG this exemption stops being correct, and the
# right fix then is to document the download, not to widen this list.
#
# The download IS documented, and until 2026-10-16 this default did not name it. The
# fallback was data/proteomes/human_reference.fasta, a path that no script writes and no
# document mentions - it occurred in exactly two tracked files, pipeline.smk and this
# comment. So a reader who ran the documented `scripts/fetch_human_proteome.py` and then
# asked for data/hard_decoys.csv still got MissingInputException, because the fetch script
# writes human_uniprot_UP000005640.fasta. Being exempt from the TRACKED check never meant
# the name was unchecked; it only meant the clone is not expected to carry the bytes.
# test_the_reference_proteome_default_is_the_path_the_fetch_script_writes now pins it.
KNOWN_UNTRACKED_DEFAULTS = {
    "reference_proteome",
    # model_path: .gitignore's model-binary block excludes every model
    # binary (models/**/*.joblib, .pkl, .pt, .pth and the models/* forms), and
    # `git ls-tree -r origin/main models/` returns ZERO .joblib. Trained
    # artifacts are deliberately not distributed in the repository. USAGE.md
    # already documents the consequence: training writes
    # models/local/rf_31feature_integrated.joblib, not the configured path,
    # and tells the reader to pass --model. This is a distribution decision,
    # not the missing-default defect this file guards, and it does not block
    # the documented dry-run (measured: exit 0 on a tracked-only export once
    # training_dataset is set).
    "model_path",
}

# config.get("<key>", "<default>") where the default looks like a path (has a / and a suffix).
_CONFIG_GET = re.compile(
    r"""config\.get\(\s*["'](?P<key>[A-Za-z0-9_]+)["']\s*,\s*["'](?P<default>[^"']*/[^"']*\.[A-Za-z0-9]+)["']\s*\)"""
)


def _inside_git_work_tree() -> bool:
    """True only if `git ls-files` here can actually answer the question we ask it.

    Everything below asks git what is TRACKED. Outside a work tree - a `git archive`
    export, a source tarball, a runner without git - `git ls-files` returns empty and
    every path reads as untracked, so the assertions would fail for a reason that has
    nothing to do with the defect they guard. An instrument that cannot measure must
    say so rather than answer anyway, so the module skips instead.

    Observed while writing this file: run inside a tracked-only export it reported
    training_dataset and binding_matrix_path as untracked, both false.
    """
    out = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


if not _inside_git_work_tree():  # pragma: no cover - environment guard
    pytest.skip(
        "not inside a git work tree, so `git ls-files` cannot report what is tracked; "
        "these assertions would be measuring the wrong thing",
        allow_module_level=True,
    )


def _tracked(rel_path: str) -> bool:
    """True if git tracks rel_path.

    Uses `git ls-files`, which reflects the INDEX rather than HEAD. That is the right
    instrument here: a path staged for deletion should fail this test before the commit
    lands, not after.
    """
    out = subprocess.run(
        ["git", "ls-files", "--", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(out.stdout.strip())


def _path_defaults() -> dict[str, str]:
    text = PIPELINE.read_text(encoding="utf-8")
    return {m.group("key"): m.group("default") for m in _CONFIG_GET.finditer(text)}


def _shipped_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_pipeline_exposes_path_defaults_to_scan():
    """Premise anchor: if the regex stops matching, every assertion below is vacuous."""
    defaults = _path_defaults()
    assert defaults, "no config.get(<key>, <path>) defaults found in pipeline.smk"
    assert "training_dataset" in defaults, (
        "pipeline.smk no longer reads training_dataset through config.get with a path "
        "default; this test's subject has moved and the test must be rewritten"
    )


def test_the_training_corpus_default_is_the_reason_this_test_exists():
    """The fallback itself is expected to remain untracked; the CONFIG must override it.

    Asserting this pins the premise. If someone later tracks the v4 corpus, the override
    stops being load-bearing and this test should be revisited rather than silently
    continuing to pass for a different reason.
    """
    fallback = _path_defaults()["training_dataset"]
    assert not _tracked(fallback), (
        f"pipeline.smk's training_dataset fallback {fallback!r} is now tracked. "
        "The defect this test guards has changed shape; re-read the module docstring."
    )


def _fetch_script_default_output(source: str | Path = FETCH_SCRIPT) -> str:
    """The DEFAULT_OUTPUT literal in scripts/fetch_human_proteome.py, read without importing.

    Parsed rather than imported: the module is a downloader, and a test must not be one
    assignment away from reaching the network. Parsed rather than grepped so that a
    commented-out assignment cannot be mistaken for the live one.

    A SHADOWED assignment is refused, not resolved. Python binds the LAST module-level
    assignment, and this helper used to return the FIRST, so two differing assignments
    would have let the reference-proteome test below compare pipeline.smk against a dead
    value. Every direct statement of the module body that targets the name is now counted:
    an Assign naming it in any target (unpacking included), an AnnAssign with or without a
    value, and an AugAssign. AssertionError is raised unless there is exactly one, and that
    one assigns a string literal to the bare name (plain, annotated or chained).

    Only direct statements of the module body are examined, so an assignment inside a def
    or class is ignored, as the CLI never reads it. The same limit means an assignment
    nested in a module-level compound statement (if, try, with, for, while, match) is NOT
    seen, and neither is a binding made by any other kind of statement, such as import,
    def, class, a for or with target, or a `:=` expression.

    `source` is the script's path (the default) or module text to parse instead.
    """
    if isinstance(source, Path):
        where, source = source.name, source.read_text(encoding="utf-8")
    else:
        where = "the given module source"
    name = "DEFAULT_OUTPUT"

    def stores(target: ast.expr) -> bool:
        return any(
            isinstance(sub, ast.Name) and sub.id == name and isinstance(sub.ctx, ast.Store)
            for sub in ast.walk(target)
        )

    found: list[tuple[ast.stmt, ast.expr | None]] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(stores(t) for t in node.targets):
            plain = any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            found.append((node, node.value if plain else None))  # unpacked: no one literal
        elif isinstance(node, ast.AnnAssign) and stores(node.target):
            found.append((node, node.value))
        elif isinstance(node, ast.AugAssign) and stores(node.target):
            found.append((node, None))  # the bound result is not the right-hand side
    if not found:
        raise AssertionError(
            f"{where} no longer assigns {name} at module level; the binding this test "
            "asserts has moved and the test must be rewritten"
        )
    if len(found) > 1:
        lines = ", ".join(str(node.lineno) for node, _ in found)
        raise AssertionError(
            f"{where} assigns {name} {len(found)} times at module level (lines {lines}). "
            "Python binds the last one; this helper refuses to guess which value the CLI "
            "uses. Leave exactly one assignment."
        )
    node, value = found[0]
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        raise AssertionError(
            f"{where} assigns {name} at line {node.lineno} by {type(node).__name__}, but not "
            "as a plain string literal, so its value cannot be read without running the module"
        )
    return value.value


# Two differing module-level assignments. A first-match reader returns the dead one.
_SHADOWED = 'DEFAULT_OUTPUT = "data/dead.fasta"\nDEFAULT_OUTPUT = "data/live.fasta"\n'


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('DEFAULT_OUTPUT = "data/one.fasta"\n', id="plain"),
        pytest.param('DEFAULT_OUTPUT: str = "data/one.fasta"\n', id="annotated"),
        pytest.param('OTHER = DEFAULT_OUTPUT = "data/one.fasta"\n', id="chained"),
        pytest.param(
            'DEFAULT_OUTPUT = "data/one.fasta"\nTABLE = {}\nTABLE[DEFAULT_OUTPUT] = "x"\n',
            id="read-in-a-target-is-not-an-assignment",
        ),
    ],
)
def test_default_output_reader_returns_a_single_string_assignment(source: str):
    assert _fetch_script_default_output(source) == "data/one.fasta"


@pytest.mark.parametrize(
    ("source", "why"),
    [
        pytest.param(_SHADOWED, "2 times", id="shadowed"),
        pytest.param(
            'DEFAULT_OUTPUT = "data/dead.fasta"\nDEFAULT_OUTPUT: str = "data/live.fasta"\n',
            "2 times",
            id="annotated-second",
        ),
        pytest.param(
            'DEFAULT_OUTPUT = "data/a.fasta"\nDEFAULT_OUTPUT += ".gz"\n', "2 times", id="augmented"
        ),
        pytest.param(
            '# DEFAULT_OUTPUT = "data/a.fasta"\nOTHER = "data/b.fasta"\n',
            "no longer assigns",
            id="none-only-a-comment",
        ),
        pytest.param("DEFAULT_OUTPUT = None\n", "not as a plain string", id="not-a-string"),
        pytest.param(
            'DEFAULT_OUTPUT = Path("data/a.fasta")\n', "not as a plain string", id="not-a-literal"
        ),
        pytest.param(
            'DEFAULT_OUTPUT, OTHER = "data/a.fasta", "data/b.fasta"\n',
            "not as a plain string",
            id="unpacked",
        ),
        pytest.param(
            'DEFAULT_OUTPUT, *REST = "data/a.fasta"\n',
            "not as a plain string",
            id="unpacked-from-one-string",
        ),
        pytest.param("DEFAULT_OUTPUT: str\n", "not as a plain string", id="bare-annotation"),
    ],
)
def test_default_output_reader_refuses_what_it_cannot_resolve(source: str, why: str):
    """Fail closed: no assignment, more than one, or one that is not a string literal."""
    with pytest.raises(AssertionError, match=why):
        _fetch_script_default_output(source)


def test_default_output_reader_ignores_assignments_outside_module_level():
    source = (
        'DEFAULT_OUTPUT = "data/live.fasta"\n'
        "def f():\n"
        '    DEFAULT_OUTPUT = "data/local.fasta"\n'
        "    return DEFAULT_OUTPUT\n"
        "class C:\n"
        '    DEFAULT_OUTPUT = "data/attribute.fasta"\n'
    )
    assert _fetch_script_default_output(source) == "data/live.fasta"


def test_the_shadowed_case_is_one_a_first_match_reader_gets_wrong(tmp_path: Path):
    """Premise for the "shadowed" case above: Python binds the LAST assignment.

    Importing the synthetic module, which is two string assignments and nothing else, gives
    data/live.fasta while its first assignment reads data/dead.fasta. A reader that returned
    the first match, as this file's helper did, would have compared pipeline.smk against a
    value the module never holds.
    """
    module_path = tmp_path / "shadowed_fetch.py"
    module_path.write_text(_SHADOWED, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("shadowed_fetch", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = ast.parse(_SHADOWED).body[0]
    assert isinstance(first, ast.Assign) and isinstance(first.value, ast.Constant)
    assert first.value.value == "data/dead.fasta"
    assert module.DEFAULT_OUTPUT == "data/live.fasta"


def test_the_reference_proteome_default_is_the_path_the_fetch_script_writes():
    """pipeline.smk must name the FASTA the documented download actually produces.

    reference_proteome is exempt from the tracked-file check above because the proteome is
    a 100 MB download rather than a repository artifact. That exemption says nothing about
    whether the NAME is right, and for a year it was not: the fallback named
    human_reference.fasta while scripts/fetch_human_proteome.py writes
    human_uniprot_UP000005640.fasta, so `snakemake data/hard_decoys.csv` failed even after
    a reader followed data/README.md exactly. An exemption from one check is not an
    exemption from all of them.
    """
    default = _path_defaults()["reference_proteome"]
    assert default == _fetch_script_default_output(), (
        f"pipeline.smk falls back to {default!r} but scripts/fetch_human_proteome.py writes "
        f"{_fetch_script_default_output()!r}. Running the documented download then asking "
        f"for data/hard_decoys.csv would fail with MissingInputException."
    )


@pytest.mark.parametrize("key", sorted(_path_defaults()))
def test_shipped_config_resolves_every_path_default_to_a_tracked_file(key: str):
    """The effective value of each path-valued key must be a file a clone actually has."""
    defaults = _path_defaults()
    effective = _shipped_config().get(key, defaults[key])

    if key in KNOWN_UNTRACKED_DEFAULTS:
        pytest.skip(f"{key} is a documented exemption; see KNOWN_UNTRACKED_DEFAULTS")

    assert _tracked(effective), (
        f"config.yaml resolves {key!r} to {effective!r}, which git does not track, so a "
        f"fresh clone cannot run the documented snakemake commands. Either set {key} in "
        f"config.yaml to a tracked path, or add it to KNOWN_UNTRACKED_DEFAULTS with the "
        f"reason it is exempt."
    )


def test_training_dataset_is_set_explicitly_not_inherited_from_the_fallback():
    """Removing the key from config.yaml must fail, even if the fallback were later fixed.

    Without this, a future change that happened to make the fallback tracked would let the
    key quietly disappear from the shipped config and the override would be lost.
    """
    config = _shipped_config()
    assert "training_dataset" in config, (
        "config.yaml no longer sets training_dataset explicitly, so pipeline.smk falls "
        "back to its own default. That fallback is not in the repository."
    )
    assert config["training_dataset"] == "data/immunogenicity_dataset_v5.csv", (
        "training_dataset moved off the v5 corpus. The rest of config.yaml declares v5 "
        "(dataset_mode, dataset_version, binding_matrix_path); they must agree."
    )


def test_the_shipped_config_agrees_with_itself_about_the_corpus_version():
    """config.yaml declared v5 everywhere except the one key the pipeline reads.

    That internal disagreement is what let the defect sit unnoticed, so it is pinned.
    """
    config = _shipped_config()
    assert config["dataset_version"] == "5.0.0"
    assert "v5" in config["dataset_mode"]
    assert "v5" in config["binding_matrix_path"]
    assert "v5" in config["training_dataset"]
