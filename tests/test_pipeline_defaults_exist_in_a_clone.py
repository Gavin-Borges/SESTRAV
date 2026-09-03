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

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "pipeline.smk"
CONFIG = REPO_ROOT / "config.yaml"

# Keys whose default is deliberately allowed to be untracked, each with the reason it is
# exempt. An exemption is recorded here so it is VISIBLE; the failure mode this whole file
# exists to catch is an untracked default that nobody knew about.
#
# reference_proteome: `pipeline.smk` falls back to data/proteomes/human_reference.fasta,
# which is not in the repository. It is exempt because it is NOT reached by the default
# target: with training_dataset set correctly, `snakemake --dry-run --cores 1` resolves to
# exit 0 on a tracked-only tree, measured. A large reference proteome is a download, not a
# repository artifact. If a future rule pulls it into the default DAG this exemption stops
# being correct, and the right fix then is to document the download, not to widen this list.
KNOWN_UNTRACKED_DEFAULTS = {
    "reference_proteome",
    # model_path: .gitignore lines 347-352 exclude every model binary
    # (models/**/*.joblib, .pkl, .pt, .pth and the models/* forms), and
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
