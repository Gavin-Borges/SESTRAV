"""scripts/stage4.py parses freeze_mode and mc_dropout, rather than calling bool() on them.

Why this test exists
--------------------
Snakemake does not hand every spelling of a boolean through as a bool. Measured
with snakemake 9.23.1, by printing the type a `params:` value of the same shape as
`rule score_immunogenicity`'s arrives with in a throwaway Snakefile:

- `--config mc_dropout=false` arrives as the str 'false', and so do `true`, `TRUE`,
  `yes` and `no`. Only the exact spellings `True` and `False` become real bools,
  and `0` / `1` become ints;
- a quoted YAML value in a config file (`mc_dropout: "false"`, `'False'`, `'no'`)
  also arrives as a str;
- the shipped `config.yaml` values, and unquoted YAML `false` / `no` / `off`,
  arrive as real bools.

Stage 4 coerced both flags with bare bool(), and bool('false') is True. So
`snakemake --config freeze_mode=false`, the obvious way to switch off the
`freeze_mode: true` that `config.yaml` ships, left freeze mode ON, and
`--config mc_dropout=false` switched MC Dropout ON. Both were silent.

The script is run here with a stand-in `snakemake` object and a stubbed
`score_immunogenicity`, so what is asserted is the value Stage 4 is actually
handed, not the shape of the source. Each accepted spelling is checked on both
flags, and anything else must raise rather than fall back to a default: a typo
in a release-grade switch is not a value to guess at.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import functions.stage4_immunogenicity_scoring as stage4_scoring

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE4 = REPO_ROOT / "scripts" / "stage4.py"
FLAGS = ("freeze_mode", "mc_dropout")


def _run_stage4(monkeypatch, tmp_path, **params) -> dict:
    """Run scripts/stage4.py as Snakemake would, returning the kwargs Stage 4 got."""
    features = tmp_path / "TEST_features.csv"
    pd.DataFrame({"peptide": ["CLGGLLTMV"]}).to_csv(features, index=False)
    captured: dict = {}

    def _score(features_df, proteome_id, **kwargs):
        captured.update(kwargs)
        return features_df, None

    monkeypatch.setattr(stage4_scoring, "score_immunogenicity", _score)
    monkeypatch.setattr(stage4_scoring, "plot_immunogenicity_scores", lambda *a, **k: None)
    fake = SimpleNamespace(
        wildcards=SimpleNamespace(proteome_id="TEST"),
        input=[str(features)],
        output=[str(tmp_path / "TEST_ranked.csv")],
        params={"model_path": "", "thresholds_path": "", "calibration_path": "", **params},
    )
    runpy.run_path(str(STAGE4), init_globals={"snakemake": fake}, run_name="stage4_under_test")
    return captured


ACCEPTED = [
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("0", False),
    ("no", False),
    ("No", False),
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("1", True),
    ("yes", True),
    ("YES", True),
    (False, False),
    (True, True),
    (0, False),
    (1, True),
]


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize(("raw", "expected"), ACCEPTED, ids=[repr(raw) for raw, _ in ACCEPTED])
def test_each_spelling_reaches_stage4_as_the_bool_it_names(monkeypatch, tmp_path, flag, raw, expected):
    captured = _run_stage4(monkeypatch, tmp_path, **{flag: raw})
    assert captured[flag] is expected, (
        f"{flag}={raw!r} reached score_immunogenicity as {captured[flag]!r}, not {expected!r}"
    )


@pytest.mark.parametrize("flag", FLAGS)
def test_an_absent_param_is_off(monkeypatch, tmp_path, flag):
    captured = _run_stage4(monkeypatch, tmp_path)
    assert captured[flag] is False


@pytest.mark.parametrize("flag", FLAGS)
def test_the_two_flags_are_parsed_independently(monkeypatch, tmp_path, flag):
    """One flag set to 'false' and the other to 'true', so neither can be read for the other."""
    other = next(name for name in FLAGS if name != flag)
    captured = _run_stage4(monkeypatch, tmp_path, **{flag: "false", other: "true"})
    assert captured[flag] is False
    assert captured[other] is True


REJECTED = ["maybe", "", "falsee", "2", "off", 2, -1, None, 0.0]


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("raw", REJECTED, ids=[repr(raw) for raw in REJECTED])
def test_anything_else_raises_instead_of_defaulting(monkeypatch, tmp_path, flag, raw):
    with pytest.raises(ValueError, match=flag) as excinfo:
        _run_stage4(monkeypatch, tmp_path, **{flag: raw})
    assert repr(raw) in str(excinfo.value), "the error does not show the value it refused"
