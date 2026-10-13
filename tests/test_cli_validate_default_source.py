"""`sestrav validate` says when its feature mode and binding matrix are defaults.

config.yaml sets feature_mode: 31 and a binding_matrix_path, and cmd_validate
takes each setting from --feature-mode / --binding-matrix first and from
_read_config second. _read_config reads config.yaml only from the installation
root and never from the working directory, so a non-editable install
(pip install ., or the Docker image) has no config there: the run falls to
validate's own fallbacks, and the "[sestrav validate] Feature mode: 21" line
reads exactly like a deliberate setting.

The two fallbacks differ, and this file states each one where it applies. The
binding matrix falls back to no matrix at all. The feature mode falls back to
"21" only where the KEY IS ABSENT - no config.yaml, PyYAML missing, an empty or
comment-only file, or a file without the key - because "21" is the default of
cmd_validate's own cfg.get. A key PRESENT but null returns None, and str(None)
makes the mode the string "None", which train_models rejects outright. That is
asserted below against the real train_models, and it is NOT a behaviour this
unit changes.

The owner ruled on 2026-10-11 to keep the behaviour and make the source
visible. Each note is keyed on the config lookup returning no value for that
key, so a configured value prints nothing even when it equals the fallback,
and an explicit flag prints nothing. Which value is used, what is forwarded to
train_models, and the exit code are unchanged in every case below; the only
difference is the extra line or lines.

train_models is replaced with a create_autospec spy on its own module, so no
model, corpus or network is touched and a forwarding change raises instead of
being absorbed.
"""

import argparse
from unittest.mock import create_autospec

import pytest

import src.cli as cli
import src.train_classifier as train_classifier

# The notes, pinned verbatim. A paraphrase in src/cli.py is a test failure.
FEATURE_NOTE = (
    "[sestrav validate] No feature_mode value came from config.yaml (sestrav reads it only "
    "from its installation root, never the working directory), so the feature mode above is "
    "not a configured setting; validate's own fallback is 21, while a feature_mode key left "
    "empty in config.yaml reaches training as the string None and is rejected there. Pass "
    "--feature-mode to set it."
)
MATRIX_NOTE = (
    "[sestrav validate] No binding_matrix_path value came from config.yaml (sestrav reads it "
    "only from its installation root, never the working directory), so this run has no "
    "binding matrix; every feature mode except 21 needs one, and --binding-matrix PATH is "
    "the only other source."
)


def _spy_train_models(monkeypatch):
    """Replace train_models on its own module; cmd_validate imports it at call time."""
    spy = create_autospec(
        train_classifier.train_models,
        return_value=(
            "rf-final",
            "xgb-final",
            {"auc_pr": 0.5, "auc_roc": 0.75, "issr_10": 0.125},
            {"auc_pr": 0.25, "auc_roc": 0.625, "issr_10": 0.0},
        ),
    )
    monkeypatch.setattr(train_classifier, "train_models", spy)
    return spy


def _validate_args(tmp_path, **overrides):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("peptide,label\nSLLMWITQV,1\n", encoding="utf-8")
    values = {
        "dataset": str(dataset),
        "binding_matrix": None,
        "model_dir": str(tmp_path / "models_out"),
        "folds": 3,
        "feature_mode": None,
        "sample_weights": False,
        "allow_overwrite": False,
        "report": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_the_notes_name_the_setting_the_lookup_and_the_flag(tmp_path, monkeypatch, capsys):
    """The installed case: no config value for either setting, no flag for either."""
    _spy_train_models(monkeypatch)
    monkeypatch.setattr(cli, "_read_config", dict)

    assert cli.cmd_validate(_validate_args(tmp_path)) == 0
    out = capsys.readouterr().out

    assert FEATURE_NOTE in out
    assert MATRIX_NOTE in out
    # The pre-existing lines are untouched, and the fallback value is still 21.
    assert "[sestrav validate] Feature mode: 21" in out
    assert "[sestrav validate] CV folds    : 3" in out


@pytest.mark.parametrize(
    "config, flags, feature_note, matrix_note",
    [
        ({}, {}, True, True),
        ({"feature_mode": 31, "binding_matrix_path": "cfg_matrix.csv"}, {}, False, False),
        ({"feature_mode": 31}, {}, False, True),
        ({"binding_matrix_path": "cfg_matrix.csv"}, {}, True, False),
        ({"feature_mode": None, "binding_matrix_path": None}, {}, True, True),
        ({"unrelated_key": 1}, {}, True, True),
        ({"feature_mode": 21}, {}, False, True),
        ({}, {"feature_mode": "33"}, False, True),
        ({}, {"binding_matrix": "flag_matrix.csv"}, True, False),
        ({}, {"feature_mode": "33", "binding_matrix": "flag_matrix.csv"}, False, False),
    ],
    ids=[
        "no-config-at-all",
        "both-configured",
        "feature-mode-configured-only",
        "binding-matrix-configured-only",
        "both-keys-present-but-null",
        "config-read-but-lacks-both-keys",
        "configured-value-equals-the-fallback",
        "explicit-feature-mode-flag",
        "explicit-binding-matrix-flag",
        "both-flags-explicit",
    ],
)
def test_each_note_tracks_its_own_lookup_and_its_own_flag(
    tmp_path, monkeypatch, capsys, config, flags, feature_note, matrix_note
):
    """A note appears exactly when that key's lookup returned no value and no flag was given.

    The configured-value-equals-the-fallback case is the discriminating one: a
    note keyed on `feature_mode == "21"` instead of on the lookup would print
    there, where config deliberately asked for 21.
    """
    spy = _spy_train_models(monkeypatch)
    monkeypatch.setattr(cli, "_read_config", lambda: dict(config))
    # An explicit --binding-matrix is existence-checked, so the flag must name a real file.
    if "binding_matrix" in flags:
        matrix = tmp_path / flags["binding_matrix"]
        matrix.write_text("peptide\n", encoding="utf-8")
        flags = dict(flags, binding_matrix=str(matrix))

    assert cli.cmd_validate(_validate_args(tmp_path, **flags)) == 0
    out = capsys.readouterr().out

    assert (FEATURE_NOTE in out) is feature_note
    assert (MATRIX_NOTE in out) is matrix_note

    # The values are unchanged by this unit: flag first, then config, then the fallback.
    expected_mode = flags.get("feature_mode") or str(config.get("feature_mode", "21"))
    expected_mode = int(expected_mode) if expected_mode.isdigit() else expected_mode
    expected_matrix = flags.get("binding_matrix") or config.get("binding_matrix_path")
    assert spy.call_args.kwargs["feature_mode"] == expected_mode
    assert spy.call_args.kwargs["binding_matrix_path"] == expected_matrix
    assert f"[sestrav validate] Feature mode: {expected_mode}" in out


@pytest.mark.parametrize(
    "contents, feature_note, matrix_note",
    [
        (None, True, True),
        ("", True, True),
        ("# only a comment\n", True, True),
        ("other_key: 1\n", True, True),
        ("feature_mode:\nbinding_matrix_path:\n", True, True),
        ("feature_mode: 33\n", False, True),
        ("feature_mode: 33\nbinding_matrix_path: cfg_matrix.csv\n", False, False),
    ],
    ids=[
        "no-config-file",
        "empty-file",
        "comment-only",
        "neither-key",
        "both-keys-null",
        "feature-mode-only",
        "both-keys",
    ],
)
def test_the_real_config_lookup_drives_the_notes(
    tmp_path, monkeypatch, capsys, contents, feature_note, matrix_note
):
    """Same cases through the real _read_config, with the installation root relocated.

    _read_config resolves config.yaml beside the installed package, so pointing
    cli.__file__ at an empty tree reproduces the non-editable install exactly,
    without a planted working-directory config ever being consulted.
    """
    _spy_train_models(monkeypatch)
    root = tmp_path / "site-packages"
    (root / "src").mkdir(parents=True)
    if contents is not None:
        (root / "config.yaml").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(root / "src" / "cli.py"))

    assert cli.cmd_validate(_validate_args(tmp_path)) == 0
    out = capsys.readouterr().out

    assert (FEATURE_NOTE in out) is feature_note
    assert (MATRIX_NOTE in out) is matrix_note


def test_this_checkout_configures_both_settings_so_neither_note_prints(
    tmp_path, monkeypatch, capsys
):
    """The editable/checkout case, read off this tree's own config.yaml.

    This is the asymmetry the notes exist for: a developer running from the
    checkout sees neither line, so the lines only ever appear where the
    settings really are fallbacks.
    """
    _spy_train_models(monkeypatch)
    cfg = cli._read_config()
    assert cfg.get("feature_mode") is not None, "this checkout's config.yaml sets feature_mode"
    assert cfg.get("binding_matrix_path") is not None, "and a binding_matrix_path"

    assert cli.cmd_validate(_validate_args(tmp_path)) == 0
    out = capsys.readouterr().out

    assert FEATURE_NOTE not in out
    assert MATRIX_NOTE not in out


@pytest.mark.parametrize(
    "config",
    [
        {"feature_mode": 0, "binding_matrix_path": ""},
        {"feature_mode": "", "binding_matrix_path": 0},
    ],
    ids=["zero-mode-and-empty-path", "empty-mode-and-zero-path"],
)
def test_a_falsy_config_value_is_still_a_configured_value(tmp_path, monkeypatch, capsys, config):
    """Each condition tests the config value with `is None`, not for falsiness.

    An empty string or a zero in config.yaml is a value somebody wrote, so
    neither note belongs: the setting did come from config. Substituting a
    falsiness test for either `is None` prints a note here, which is wrong
    about where the value came from, so this case is what separates the two
    spellings. The values forwarded are unchanged either way, which is why no
    output assertion below depends on the note.
    """
    spy = _spy_train_models(monkeypatch)
    monkeypatch.setattr(cli, "_read_config", lambda: dict(config))

    assert cli.cmd_validate(_validate_args(tmp_path)) == 0
    out = capsys.readouterr().out

    assert FEATURE_NOTE not in out
    assert MATRIX_NOTE not in out
    expected_mode = str(config["feature_mode"])
    expected_mode = int(expected_mode) if expected_mode.isdigit() else expected_mode
    assert spy.call_args.kwargs["feature_mode"] == expected_mode
    assert spy.call_args.kwargs["binding_matrix_path"] == config["binding_matrix_path"]


@pytest.mark.parametrize(
    "flags, feature_note, matrix_note",
    [
        ({"feature_mode": ""}, True, True),
        ({"binding_matrix": ""}, True, True),
    ],
    ids=["empty-feature-mode-flag", "empty-binding-matrix-flag"],
)
def test_an_empty_flag_is_not_a_flag_so_its_note_still_prints(
    tmp_path, monkeypatch, capsys, flags, feature_note, matrix_note
):
    """The flag halves are falsy tests, and that is the behaviour being pinned.

    `args.feature_mode or ...` and `args.binding_matrix or ...` both fall
    through on an empty string, so an empty flag does not supply the setting
    and its note must still print. Spelling either flag half as `is None`
    would silence the note while the value is still a fallback, which is the
    mirror image of the case above.
    """
    _spy_train_models(monkeypatch)
    monkeypatch.setattr(cli, "_read_config", dict)

    assert cli.cmd_validate(_validate_args(tmp_path, **flags)) == 0
    out = capsys.readouterr().out

    assert (FEATURE_NOTE in out) is feature_note
    assert (MATRIX_NOTE in out) is matrix_note


def test_both_notes_are_printed_before_train_models_is_called(tmp_path, monkeypatch, capsys):
    """Ordering is load-bearing: each note says "the feature mode above".

    Printed after the train_models call, the notes would follow that call's own
    output, so "above" would name the wrong line - and a run that raises in
    training would print no note at all, which is exactly the run that most
    needs one. Both halves are asserted: the position against a marker the
    stubbed trainer prints, and survival of a raising trainer.
    """
    marker = "[spy] train_models entered"

    def _spy_prints_then_returns(*args, **kwargs):
        print(marker)
        return (
            "rf-final",
            "xgb-final",
            {"auc_pr": 0.5, "auc_roc": 0.75, "issr_10": 0.125},
            {"auc_pr": 0.25, "auc_roc": 0.625, "issr_10": 0.0},
        )

    spy = create_autospec(train_classifier.train_models, side_effect=_spy_prints_then_returns)
    monkeypatch.setattr(train_classifier, "train_models", spy)
    monkeypatch.setattr(cli, "_read_config", dict)

    assert cli.cmd_validate(_validate_args(tmp_path)) == 0
    out = capsys.readouterr().out

    assert marker in out
    assert out.index(FEATURE_NOTE) < out.index(marker)
    assert out.index(MATRIX_NOTE) < out.index(marker)


def test_the_notes_survive_a_train_models_that_raises(tmp_path, monkeypatch, capsys):
    """The run that needs the notes most is the one that dies in training.

    A null feature_mode key is exactly that run: cmd_validate forwards the
    string "None" and train_models rejects it (see the test below). The notes
    must already be on stdout when that happens.
    """
    spy = create_autospec(
        train_classifier.train_models,
        side_effect=ValueError("Unknown feature_mode 'None'"),
    )
    monkeypatch.setattr(train_classifier, "train_models", spy)
    monkeypatch.setattr(cli, "_read_config", lambda: {"feature_mode": None})

    with pytest.raises(ValueError, match="Unknown feature_mode"):
        cli.cmd_validate(_validate_args(tmp_path))
    out = capsys.readouterr().out

    assert FEATURE_NOTE in out
    assert MATRIX_NOTE in out
    assert spy.call_args.kwargs["feature_mode"] == "None"


def test_the_real_train_models_rejects_the_string_none(tmp_path):
    """Why the feature note does not call every un-configured mode "21".

    A `feature_mode:` key present but null returns None from cfg.get, so
    cmd_validate's `str(cfg.get("feature_mode", "21"))` yields "None" rather
    than the "21" that an ABSENT key yields. The real train_models validates
    feature_mode against RECOGNIZED_FEATURE_MODES before touching any path, so
    this call raises without reading the dataset or writing to model_dir.
    Asserted against the real function, not a stub, because it is the claim the
    note and the CHANGELOG entry rest on.
    """
    with pytest.raises(ValueError, match="Unknown feature_mode 'None'"):
        train_classifier.train_models(
            data_path=str(tmp_path / "never-read.csv"),
            model_dir=str(tmp_path / "never-written"),
            feature_mode="None",
        )
    assert not (tmp_path / "never-written").exists()
    assert "None" not in train_classifier.RECOGNIZED_FEATURE_MODES


def test_the_notes_print_below_the_three_lines_they_refer_to(tmp_path, monkeypatch, capsys):
    """The feature note says "the feature mode above", so its POSITION is part of the claim.

    Moved above the three pre-existing "[sestrav validate]" prints, the block is
    still before train_models, so the ordering test above still passes, while the
    word "above" would name a line that has not been printed yet. Each note is
    asserted to follow all three of those lines, which is what makes that word true.
    """
    _spy_train_models(monkeypatch)
    monkeypatch.setattr(cli, "_read_config", dict)
    args = _validate_args(tmp_path)

    assert cli.cmd_validate(args) == 0
    out = capsys.readouterr().out

    for line in (
        f"[sestrav validate] Dataset     : {args.dataset}",
        "[sestrav validate] Feature mode: 21",
        f"[sestrav validate] CV folds    : {args.folds}",
    ):
        assert line in out
        assert out.index(line) < out.index(FEATURE_NOTE)
        assert out.index(line) < out.index(MATRIX_NOTE)


def test_both_validate_binding_matrix_help_strings_state_the_requirement(capsys):
    """Two parsers advertise --binding-matrix for validate, and both had read "mode 30+".

    "30+" is wrong for mode 10: train_models raises "--binding-matrix is required
    for feature-mode 10" when it is None. Both strings are pinned because only one
    of the two is reachable from "sestrav validate --help", so a fix applied to one
    is easy to miss on the other. src/train_classifier.py's own --binding-matrix
    help is pinned twice in tests/test_binding_matrix_requirement_wording.py: once
    on its own, and once in that file's four-carrier agreement test.
    """
    standalone = cli._build_validate_parser()
    action = next(a for a in standalone._actions if "--binding-matrix" in a.option_strings)
    assert (
        action.help
        == "Path to peptide binding matrix CSV (required for every feature mode except 21)"
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["validate", "--help"])
    assert exit_info.value.code == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert "Binding matrix CSV (required for every feature mode except 21)" in rendered
    assert "30+" not in rendered
