"""The Snakemake scoring rule forwards every config key Stage 4 scores with.

Why this test exists
--------------------
`rule score_immunogenicity` (pipeline.smk) declares the config keys it reads as
`params`, and `scripts/stage4.py` passes them to `score_immunogenicity`. Two keys
were declared in `config.yaml`, read by `pipeline.py`, and dropped here: setting
`calibration_path` had no effect under Snakemake (calibration fell back to a
calibrator found beside the model), and `mc_dropout: true` produced no MC Dropout
pass at all, though `docs/model_cards/ann_30feature.md` tells a reader to enable it
by setting exactly that key in `config.yaml` (to-do PC-12). `thresholds_path` had
the same defect until it was fixed, and
`tests/test_pipeline_forwards_freeze_mode.py` pins the mirror case, where
`pipeline.py` was the side that dropped a key.

Pinned here, by reading the sources rather than running Snakemake, which needs a
real DAG and a conda environment:

- every key the rule declares as a param is passed on by `scripts/stage4.py`;
- the five scoring keys `config.yaml` carries reach `score_immunogenicity`;
- each is passed to the parameter of the same name, so none is crossed;
- `mc_dropout` and `freeze_mode` go through the script's string-aware
  `_parse_flag`, not bare bool(), and the two path keys turn an empty string into
  None, which is what an unset config key becomes. Bare bool() made
  `--config mc_dropout=false` truthy, because Snakemake delivers that as the str
  'false'; `tests/test_stage4_parses_boolean_params.py` runs the script and pins
  what each spelling actually reaches Stage 4 as.

This is a static check: it says the keys are wired, not that Stage 4 honours them.
`tests/test_stage4_scoring.py::test_pytorch_branch_mc_dropout` covers the
behaviour at the far end.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMK = REPO_ROOT / "pipeline.smk"
STAGE4 = REPO_ROOT / "scripts" / "stage4.py"
RULE = "score_immunogenicity"

# The scoring keys config.yaml carries. Each must reach score_immunogenicity.
CONFIG_SCORING_KEYS = {"model_path", "thresholds_path", "calibration_path", "freeze_mode", "mc_dropout"}


def _rule_params(rule: str, with_default: bool = False):
    """{param name: the config key it reads} for one rule's params block.

    With ``with_default``, the value is ``(config key, default as written)`` instead.
    """
    text = SMK.read_text(encoding="utf-8")
    start = text.index(f"\nrule {rule}:")
    end = text.find("\nrule ", start + 1)
    body = text[start : end if end != -1 else len(text)]
    params = body.index("\n    params:")
    stop = min(
        (i for i in (body.find(f"\n    {k}:", params) for k in ("log", "benchmark", "conda", "script", "shell")) if i != -1),
        default=len(body),
    )
    block = body[params:stop]
    found = {
        m.group(1): (m.group(2), (m.group(3) or "").strip())
        for m in re.finditer(
            r"^\s+(\w+)\s*=\s*config\.get\(\s*\"(\w+)\"\s*(?:,([^)]*))?\)", block, re.MULTILINE
        )
    }
    return found if with_default else {name: key for name, (key, _) in found.items()}


def _stage4_call() -> ast.Call:
    tree = ast.parse(STAGE4.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "score_immunogenicity":
            return node
    raise AssertionError("scripts/stage4.py does not call score_immunogenicity")


def _params_read_by_stage4() -> set[str]:
    """Every name fetched from snakemake.params in scripts/stage4.py."""
    tree = ast.parse(STAGE4.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "params"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names.add(node.args[0].value)
    return names


def test_the_rule_declares_every_scoring_key_config_carries() -> None:
    params = _rule_params(RULE)
    assert params, "no config.get params parsed from the rule"
    missing = CONFIG_SCORING_KEYS - set(params.values())
    assert not missing, f"{RULE} does not read {sorted(missing)} from config"
    for name, key in params.items():
        assert name == key, f"param {name} reads config key {key}; keep the names equal"


def test_stage4_reads_every_param_the_rule_declares() -> None:
    declared = set(_rule_params(RULE))
    read = _params_read_by_stage4()
    assert not declared - read, f"scripts/stage4.py ignores params {sorted(declared - read)}"


def test_every_scoring_key_reaches_score_immunogenicity_uncrossed() -> None:
    """Each key is passed to the keyword of its own name, so none is wired to another."""
    call = _stage4_call()
    passed = {kw.arg: kw.value for kw in call.keywords}
    for key in CONFIG_SCORING_KEYS:
        assert key in passed, f"score_immunogenicity is not given {key}"
        source = ast.unparse(passed[key])
        assert re.search(rf"\b{key}\b", source), f"{key}= is passed {source}, which does not derive from it"


def test_an_absent_config_key_defaults_to_off() -> None:
    """The rule's own default decides when config omits a key, as the smoke fixture does."""
    for name, (key, default) in _rule_params(RULE, with_default=True).items():
        assert default in {'""', "False"}, f"{name} defaults to {default!r}; it must default to off"


def test_no_assignment_discards_a_forwarded_value() -> None:
    """Every assignment to a forwarded name must still derive from the config read.

    `model_path = None` under `if model_path == "":` is normalization and is allowed,
    because its own guard tests the name. A bare `calibration_path = None` between the
    read and the call is not: it would silently drop what config supplied, and the
    keyword-name check above cannot see it.
    """
    tree = ast.parse(STAGE4.read_text(encoding="utf-8"))

    def walk(node: ast.AST, guards: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = guards
            if isinstance(child, ast.If):
                inner = guards + (ast.unparse(child.test),)
            if isinstance(child, ast.Assign):
                rhs = ast.unparse(child.value)
                for target in child.targets:
                    name = getattr(target, "id", None)
                    if name not in CONFIG_SCORING_KEYS:
                        continue
                    derived = "snakemake.params" in rhs or re.search(rf"\b{name}\b", rhs)
                    guarded = any(re.search(rf"\b{name}\b", g) for g in inner)
                    assert derived or guarded, (
                        f"{name} = {rhs} neither derives from the config read nor normalizes it"
                    )
            walk(child, inner)

    walk(tree, ())


def test_the_flags_are_coerced_and_empty_paths_become_none() -> None:
    """An unset config key arrives as "" or False; Stage 4 must normalize before passing."""
    source = STAGE4.read_text(encoding="utf-8")
    for flag in ("freeze_mode", "mc_dropout"):
        assert re.search(rf"{flag}\s*=\s*_parse_flag\(\s*\"{flag}\"", source), (
            f"{flag} is not parsed by _parse_flag under its own name"
        )
        assert not re.search(rf"{flag}\s*=\s*bool\(", source), (
            f"{flag} is coerced with bare bool(), which makes the str 'false' True"
        )
    for path_key in ("thresholds_path", "calibration_path"):
        assert re.search(rf"{path_key}\s*=.*\bor None\b", source), (
            f"{path_key} does not turn an empty config value into None"
        )
