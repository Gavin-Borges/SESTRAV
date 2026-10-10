"""SECURITY.md's CI gate map must name every security job that can report a finding.

README.md and CONTRIBUTING.md both defer to that table as the authoritative list, so
a job missing from it is a gate a reader is told does not exist. The map omitted the
`Custom Secret Pattern Scan` job in security.yml outright, and the paragraph under it
counted "Three" advisory jobs that turn their own run red while the table itself
already described more than three. A count word is checked by nobody; the job names
can be checked against the workflows they come from, which is what this test does.

The check context of a job is its `name:`, or its job id when it has none, which is
also the string a branch ruleset's required_status_checks entry has to match.

The review gate is covered too. pr-review-check.yml returns success without any
review for the bot logins in its TRUSTED_BOTS list, and SECURITY.md's description of
that required check has to name each of them, or a reader concludes bot pull requests
are human-reviewed.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_MD = REPO_ROOT / "SECURITY.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Workflows whose every job reports a security or supply-chain finding. Named
# explicitly rather than globbed: a glob would also pull in ci.yml's lint job or the
# docs citation gates, which are not security gates and are covered elsewhere.
SECURITY_WORKFLOWS = (
    "security.yml",
    "dependency-review.yml",
    "pii_scan.yml",
    "dependency-lockfile-check.yml",
    "lock_manifest_coverage.yml",
    "scorecard.yml",
    "dismissal_justifications_advisory.yml",
)


def _gate_map_section() -> str:
    text = SECURITY_MD.read_text(encoding="utf-8")
    start = text.index("### CI gate map")
    end = text.find("\n### ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _check_contexts() -> list[tuple[str, str]]:
    contexts = []
    for workflow in SECURITY_WORKFLOWS:
        doc = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
        for job_id, job in doc["jobs"].items():
            contexts.append((workflow, job.get("name") or job_id))
    return contexts


def test_gate_map_names_every_security_job() -> None:
    contexts = _check_contexts()
    # Anti-vacuity: security.yml alone carries seven jobs, so an empty or truncated
    # parse must not read as "nothing missing".
    assert len(contexts) >= len(SECURITY_WORKFLOWS) + 6, contexts
    section = _gate_map_section()
    missing = [
        f"{workflow}: {context}" for workflow, context in contexts if f"`{context}`" not in section
    ]
    assert not missing, (
        f"SECURITY.md's CI gate map does not name these jobs by their check context: {missing}"
    )


def test_review_gate_bot_bypass_is_documented() -> None:
    workflow = (WORKFLOWS / "pr-review-check.yml").read_text(encoding="utf-8")
    # Up to `];`, not the first `]`: every entry ends in `[bot]`.
    match = re.search(r"TRUSTED_BOTS\s*=\s*\[(.*?)\];", workflow)
    assert match, "pr-review-check.yml no longer declares TRUSTED_BOTS; update this test"
    bots = re.findall(r"'([^']+)'", match.group(1))
    assert bots, "TRUSTED_BOTS parsed as empty"
    # Search only the review-gate paragraph that names TRUSTED_BOTS, not the whole
    # file: `dependabot[bot]` also appears in the gate map's lockfile-freshness row,
    # so a whole-file search would stay green with the bypass text deleted.
    text = SECURITY_MD.read_text(encoding="utf-8").replace("\r\n", "\n")
    paragraphs = [p for p in text.split("\n\n") if "`TRUSTED_BOTS`" in p]
    assert len(paragraphs) == 1, (
        "expected exactly one SECURITY.md paragraph describing the review gate's "
        f"`TRUSTED_BOTS` bypass, found {len(paragraphs)}"
    )
    missing = [bot for bot in bots if f"`{bot}`" not in paragraphs[0]]
    assert not missing, (
        "pr-review-check.yml passes the required `Require human review` check without "
        f"a review for these authors, and SECURITY.md does not say so: {missing}"
    )
    # And the other direction: a bot the paragraph names as exempt must still be in the
    # list, or a reader is told a bypass exists that the workflow no longer grants.
    named = set(re.findall(r"`([A-Za-z0-9][A-Za-z0-9-]*\[bot\])`", paragraphs[0]))
    stale = sorted(named - set(bots))
    assert not stale, (
        "SECURITY.md says these bots bypass `Require human review`, but "
        f"pr-review-check.yml's TRUSTED_BOTS no longer lists them: {stale}"
    )
