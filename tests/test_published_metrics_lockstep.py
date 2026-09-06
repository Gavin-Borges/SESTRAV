"""CI-visible lockstep for the five duplicated contamination-disclosure literals.

B2c Option 3 (structural_fix_plans_2026-09-01.md): do not extract the numbers
out of api/main.py or app/demo.py, because the live cv.mode31.auc_pr_pooled
claim binds those files by presence. A tracked test is the half of that plan
that CI can see; the harness half lives in gitignored claims_manifest.toml.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Five values duplicated between /model-card and the demo expander.
# Corpus counts (35,597 / 51,185 / 5,000) are api-only and are not in this set.
_LOCKSTEP_TOKENS = ("0.6055", "0.658", "71.1", "0.8347", "0.6092")


def test_contamination_disclosure_literals_match_between_api_and_demo() -> None:
    api = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    demo = (REPO_ROOT / "app" / "demo.py").read_text(encoding="utf-8")
    missing_api = [tok for tok in _LOCKSTEP_TOKENS if tok not in api]
    missing_demo = [tok for tok in _LOCKSTEP_TOKENS if tok not in demo]
    assert not missing_api, f"api/main.py missing {missing_api}"
    assert not missing_demo, f"app/demo.py missing {missing_demo}"
