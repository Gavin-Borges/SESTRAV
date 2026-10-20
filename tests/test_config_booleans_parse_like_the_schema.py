"""config.yaml's boolean keys mean what the config schema says, at every reader.

Why this test exists
--------------------
`src/core/config.py` declares `freeze_mode: bool` on `SestravConfig` and
`require_checksum_match_in_freeze_mode: bool` on `DatasetGovernanceConfig`. pydantic
gives those fields a defined vocabulary: in any case, `true`, `on`, `yes`, `1`, `t` and
`y` are True and `false`, `off`, `no`, `0`, `f` and `n` are False, and anything else is
refused. `pipeline.py` reads the flags through that schema.

Three other readers bypassed it and re-read the raw YAML, testing the value for
truthiness:

- `src/cli.py` (`sestrav predict`) set `freeze_mode = bool(configured)`;
- `src/data_curation_qc.py` and `scripts/data_qc_gate.py` gated the dataset checksum on
  `config.get("freeze_mode") and gov.get("require_checksum_match_in_freeze_mode")`.

An unquoted YAML `false`, `no` or `off` arrives from `yaml.safe_load` as a real bool and
was fine. A QUOTED one arrives as a string, and `bool("false")` is True, so
`freeze_mode: "false"` turned freeze mode ON at those three readers while the schema read
it as off, and `require_checksum_match_in_freeze_mode: "false"` enforced the checksum.
Both failed toward the stricter mode, so nothing unsafe passed; but the same config.yaml
meant two different things depending on which entry point read it, and a run the
operator had switched off could still be refused.

What is asserted
----------------
- `src.config_flags.config_bool` agrees with the schema's own fields on every spelling
  probed, including the edges pydantic refuses (padding whitespace, the strings "1.0" and
  "0.0", "none", 2, 0.5). This is checked against `SestravConfig` and
  `DatasetGovernanceConfig` themselves, so a later change to either field's type is caught
  here. It needs pydantic, which only the `dev`, `api` and `demo` extras install, which is
  exactly why `config_bool` reimplements the vocabulary instead of importing pydantic:
  `sestrav predict` must keep working on a core-only install.
- A missing or null key (`None`) is the reader's default, as it always was. This is the
  one deliberate difference from the schema, whose fields are required.
- At each of the three readers, a quoted `"false"` now switches the flag off, a quoted
  `"true"` still switches it on, and a value that is not a boolean is refused with the key
  named, rather than read as on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every outcome pydantic was measured to give, plus a non-ASCII spelling.
PROBES = [
    "true",
    "True",
    "TRUE",
    "on",
    "ON",
    "yes",
    "Yes",
    "1",
    "t",
    "T",
    "y",
    "Y",
    "false",
    "False",
    "FALSE",
    "off",
    "OFF",
    "no",
    "No",
    "0",
    "f",
    "F",
    "n",
    "N",
    " true",
    "true ",
    "\ttrue",
    "tru",
    "yess",
    "2",
    "-1",
    "1.0",
    "0.0",
    "",
    "  ",
    "none",
    "None",
    "null",
    # "true" with a combining acute accent. Built with chr() because the encoding gate
    # reads a string literal's VALUE, so an escape sequence would be flagged like a glyph.
    "true" + chr(0x0301),
    True,
    False,
    0,
    1,
    2,
    -1,
    0.0,
    1.0,
    0.5,
]


def _config_bool():
    """Imported per test: a module-level import of a missing module would be a collection
    error, which runs no test at all."""
    from src.config_flags import config_bool

    return config_bool


def _outcome(fn, value):
    try:
        return ("ok", fn(value))
    except ValueError:
        return ("refused", None)


# ---------------------------------------------------------------------------
# config_bool itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field"),
    [
        ("SestravConfig", "freeze_mode"),
        ("DatasetGovernanceConfig", "require_checksum_match_in_freeze_mode"),
    ],
)
def test_config_bool_agrees_with_the_schema_on_every_spelling(model_name, field):
    pydantic = pytest.importorskip("pydantic", reason="the schema is pydantic")
    from src.core import config as schema

    annotation = getattr(schema, model_name).model_fields[field].annotation
    adapter = pydantic.TypeAdapter(annotation)
    config_bool = _config_bool()

    def through_schema(value):
        try:
            return adapter.validate_python(value)
        except pydantic.ValidationError as exc:
            raise ValueError(str(exc)) from exc

    disagreements = [
        (value, _outcome(through_schema, value), _outcome(lambda v: config_bool(field, v), value))
        for value in PROBES
        if _outcome(through_schema, value) != _outcome(lambda v: config_bool(field, v), value)
    ]
    assert not disagreements, f"config_bool disagrees with {model_name}.{field} on: {disagreements}"


def test_a_missing_or_null_key_is_the_readers_default():
    config_bool = _config_bool()
    assert config_bool("freeze_mode", None) is False
    assert config_bool("freeze_mode", None, default=True) is True


@pytest.mark.parametrize("value", ["maybe", "", 2, 0.5, " false"])
def test_a_value_that_is_not_a_boolean_is_refused_naming_key_and_value(value):
    config_bool = _config_bool()
    with pytest.raises(ValueError) as excinfo:
        config_bool("freeze_mode", value)
    message = str(excinfo.value)
    assert "freeze_mode" in message and repr(value) in message, message


def test_config_bool_does_not_import_pydantic():
    """A core-only install has no pydantic; sestrav predict must not need it."""
    script = (
        "import sys; sys.modules['pydantic'] = None\n"
        "from src.config_flags import config_bool\n"
        "assert config_bool('k', 'false') is False\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr


# ---------------------------------------------------------------------------
# src/cli.py: sestrav predict
# ---------------------------------------------------------------------------


def _stub_stages(monkeypatch, captured):
    """As tests/test_cli_predict_forwards_freeze_mode.py: every stage stubbed."""
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    import functions.stage1_peptide_generation as s1
    import functions.stage2_mhc_binding_prediction as s2
    import functions.stage3_tcr_feature_extraction as s3
    import functions.stage4_immunogenicity_scoring as s4

    monkeypatch.setattr(s1, "generate_peptides", lambda *a, **k: frame)
    monkeypatch.setattr(s2, "predict_binding", lambda *a, **k: frame)
    monkeypatch.setattr(s3, "extract_tcr_features", lambda *a, **k: frame)

    def _score(features_df, proteome_id, **kwargs):
        captured.update(kwargs)
        return frame, None

    monkeypatch.setattr(s4, "score_immunogenicity", _score)
    from src import cli

    monkeypatch.setattr(cli, "_mhcflurry_model_data_path", lambda: "stubbed")


def _predict_args(tmp_path):
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">x\nCLGGLLTMV\n", encoding="utf-8")
    model = tmp_path / "m.joblib"
    model.write_bytes(b"stub")
    # With freeze mode on, cmd_predict refuses to start without a conformal calibrator.
    (tmp_path / "conformal_calibrator.joblib").write_bytes(b"stub")
    return argparse.Namespace(
        fasta=str(fasta),
        model=str(model),
        output=str(tmp_path / "out"),
        alleles=None,
        lengths=None,
        conformal=True,
        conformal_calibrator=None,
        virus=None,
        per_virus_calibration_dir=None,
        freeze_mode=None,
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("false", False), ("FALSE", False), ("off", False), ("true", True), ("yes", True)],
)
def test_sestrav_predict_reads_a_quoted_freeze_mode_as_the_schema_does(
    monkeypatch, tmp_path, configured, expected
):
    from src import cli

    captured: dict = {}
    _stub_stages(monkeypatch, captured)
    monkeypatch.setattr(cli, "_read_config", lambda: {"freeze_mode": configured})
    cli.cmd_predict(_predict_args(tmp_path))
    assert captured["freeze_mode"] is expected, (
        f"config.yaml freeze_mode: {configured!r} reached Stage 4 as {captured['freeze_mode']!r}"
    )


def test_sestrav_predict_refuses_a_freeze_mode_that_is_not_a_boolean(monkeypatch, tmp_path):
    from src import cli

    _stub_stages(monkeypatch, {})
    monkeypatch.setattr(cli, "_read_config", lambda: {"freeze_mode": "maybe"})
    with pytest.raises(ValueError, match="freeze_mode"):
        cli.cmd_predict(_predict_args(tmp_path))


def test_sestrav_predict_still_defaults_off_when_config_has_no_value(monkeypatch, tmp_path, capsys):
    from src import cli

    captured: dict = {}
    _stub_stages(monkeypatch, captured)
    monkeypatch.setattr(cli, "_read_config", lambda: {})
    cli.cmd_predict(_predict_args(tmp_path))
    assert captured["freeze_mode"] is False
    assert "No freeze_mode value came from config.yaml" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# src/data_curation_qc.py and scripts/data_qc_gate.py: the dataset checksum
# ---------------------------------------------------------------------------

MISMATCHED_CHECKSUM = "0" * 64


def _curation_dataset(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"peptide": ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"], "label": [1, 0, 1]}).to_csv(
        path, index=False
    )
    return path


def _gate_dataset(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "ACDEFGHIKL", "LMNPQRSTVY"],
            "label": [1, 0, 1, 0, 1],
            "allele": ["HLA-A*02:01", "HLA-B*08:01", "HLA-A*02:01", None, "HLA-A*11:01"],
        }
    ).to_csv(path, index=False)
    return path


def _config(tmp_path, freeze_mode, require):
    """freeze_mode and require are YAML scalars, written verbatim, so quoting is exact."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset_governance:\n"
        "  qc_thresholds:\n"
        "    min_peptide_yield: 2\n"
        "    max_conflict_ratio: 0.15\n"
        "    max_null_allele_fraction: 0.50\n"
        "    class_ratio_bounds: [0.1, 10.0]\n"
        f"  require_checksum_match_in_freeze_mode: {require}\n"
        "  provenance:\n"
        f'    checksum: "{MISMATCHED_CHECKSUM}"\n'
        f"freeze_mode: {freeze_mode}\n",
        encoding="utf-8",
    )
    return path


def _run(script, dataset_flag, dataset, config):
    result = subprocess.run(
        [sys.executable, script, dataset_flag, str(dataset), "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


SCRIPTS = [
    pytest.param(
        "src/data_curation_qc.py", "--check-dataset", _curation_dataset, id="data_curation_qc"
    ),
    pytest.param("scripts/data_qc_gate.py", "--dataset", _gate_dataset, id="data_qc_gate"),
]


@pytest.mark.parametrize(("script", "flag", "dataset"), SCRIPTS)
def test_a_quoted_false_freeze_mode_does_not_enforce_the_checksum(tmp_path, script, flag, dataset):
    rc, output = _run(script, flag, dataset(tmp_path), _config(tmp_path, '"false"', "true"))
    assert rc == 0, f'freeze_mode: "false" still enforced the dataset checksum:\n{output}'


@pytest.mark.parametrize(("script", "flag", "dataset"), SCRIPTS)
def test_a_quoted_false_requirement_does_not_enforce_the_checksum(tmp_path, script, flag, dataset):
    rc, output = _run(script, flag, dataset(tmp_path), _config(tmp_path, "true", '"false"'))
    assert rc == 0, (
        f'require_checksum_match_in_freeze_mode: "false" still enforced the checksum:\n{output}'
    )


@pytest.mark.parametrize(("script", "flag", "dataset"), SCRIPTS)
def test_a_quoted_true_still_enforces_the_checksum(tmp_path, script, flag, dataset):
    """The other direction, so a fix that simply ignored quoted values would fail."""
    rc, output = _run(script, flag, dataset(tmp_path), _config(tmp_path, '"true"', '"yes"'))
    assert rc != 0, f"a quoted true freeze_mode stopped enforcing the checksum:\n{output}"
    assert "does not match" in output, output


@pytest.mark.parametrize(("script", "flag", "dataset"), SCRIPTS)
def test_a_freeze_mode_that_is_not_a_boolean_stops_the_run(tmp_path, script, flag, dataset):
    """The requirement is an UNQUOTED false, a real bool, so nothing else can fail the run.

    With `require: true` a first draft of this test passed against the unfixed gate for
    the wrong reason: "maybe" is truthy, so the mismatched checksum was enforced and the
    run failed, and "maybe" appeared in the configuration the gate logs. Here the old code
    passes the run, and only the refusal can fail it.
    """
    rc, output = _run(script, flag, dataset(tmp_path), _config(tmp_path, '"maybe"', "false"))
    assert rc != 0, f'freeze_mode: "maybe" was accepted:\n{output}'
    assert "is not a boolean" in output and "freeze_mode" in output, output
