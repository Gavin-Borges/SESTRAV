"""The tag-only Docker publish must not read the shared GitHub Actions cache.

`docker.yml` triggers on `push: tags` and on `workflow_dispatch`. The Actions cache
is scoped so that a run on a tag ref restores entries SAVED BY DEFAULT-BRANCH RUNS,
so with `cache-from: type=gha` configured any code executing in a main-branch job -
a compromised build dependency, for instance - could plant BuildKit layers that the
release build imports, bakes into the published image, and has the `provenance` and
`sbom` attestations sign as genuine. Fork pull requests get an isolated cache scope
and were never the exposure, which is why the usual "untrusted PR" reasoning does
not cover this one.

These guards are content guards on the workflow: there is no way to exercise the
exposure locally, and at the time they were written `docker.yml` had never run.
"""

from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "docker.yml"
BUILD_ACTION = "docker/build-push-action@"


def _document() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(document: dict) -> dict:
    """The `on:` mapping.

    YAML 1.1 reads a bare `on` as the boolean True, which is what `yaml.safe_load`
    implements, so the key is looked up both ways rather than assumed.
    """
    for key in ("on", True):
        if key in document:
            return document[key]
    raise AssertionError(f"no `on:` key in {WORKFLOW.name}; the guard cannot run")


def _build_steps(document: dict) -> list[dict]:
    return [
        step
        for job in document["jobs"].values()
        for step in job.get("steps", [])
        if BUILD_ACTION in str(step.get("uses", ""))
    ]


def test_the_image_build_step_declares_no_buildkit_cache() -> None:
    """Parsed from the workflow, so any cache backend is caught, not just type=gha."""
    steps = _build_steps(_document())
    assert steps, f"no {BUILD_ACTION} step in {WORKFLOW.name}; this guard has gone vacuous"
    for step in steps:
        inputs = step.get("with", {}) or {}
        cache_keys = sorted(key for key in inputs if str(key).startswith("cache"))
        assert not cache_keys, (
            f"step {step.get('name')!r} configures {cache_keys}; the release build must "
            "import no layers from the shared Actions cache"
        )


def test_the_workflow_text_names_no_gha_cache_backend() -> None:
    """Text-level backstop, in case a cache is smuggled past the parser.

    It would read a `cache-from` hidden in a `run:` script or a commented-out key
    that a later edit might uncomment, neither of which the parsed check above sees.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    # Positive control: proves the file really was read and really is this workflow.
    assert BUILD_ACTION in text, f"{WORKFLOW} does not look like the Docker workflow"
    assert "type=gha" not in text, (
        f"{WORKFLOW.name} still names the type=gha cache backend; a tag run would "
        "restore layers saved by default-branch runs"
    )


def test_the_publish_is_still_tag_only_so_the_cache_scope_still_matters() -> None:
    """The premise the guards above rest on.

    If `docker.yml` ever also builds on pushes to the default branch, the reasoning
    recorded here changes shape and has to be re-derived rather than inherited.
    """
    document = _document()
    push = _triggers(document).get("push")
    assert push is not None, "docker.yml no longer triggers on push; re-derive the rationale"
    assert set(push) == {"tags"}, f"push trigger is no longer tag-only: {sorted(push)}"

    steps = _build_steps(document)
    assert steps, f"no {BUILD_ACTION} step in {WORKFLOW.name}; this guard has gone vacuous"
    for step in steps:
        inputs = step.get("with", {}) or {}
        assert inputs.get("push") == "${{ github.event_name == 'push' }}", inputs.get("push")
