"""The binding matrix is required for every feature mode except 21, and the carriers say so.

`train_models`' feature-selection chain gives every recognised feature mode except 21 its
own branch, and each of those branches raises `--binding-matrix is required for
feature-mode <mode>` when `binding_matrix_path` is None. 21 is the chain's final `else`,
which builds its features with `include_binding=False`, so it is the one mode that trains
with no matrix at all.

Two carriers described a narrower rule. `src/train_classifier.py`'s `--binding-matrix` help
read "required for --feature-mode 30+", which is wrong for mode 10, and `USAGE.md`'s
`sestrav validate` table read "Required for `--feature-mode 30` or `31`", which is wrong for
the other eight: 10, 33, 35, 50, 51, 166, `30_esm` and `30_graph`. Both now carry the phrase
`src/cli.py`'s two `validate` help strings already use for the same requirement, "required
for every feature mode except 21", which is why this file also asserts that the four
carriers hold one phrase between them rather than the three different sentences they held
before.

The requirement is EXERCISED here rather than read off the module's source text. A
twelve-row synthetic corpus is enough, because each raise fires in the dispatch just after
that corpus is loaded: the ten raise cases fit no model and write no artifact, so the
mode-21 case is the only one of the eleven `train_models` cases here that trains, on two
folds of a ten-record training pool. No timing figure is pinned here, because a stopwatch
reading is a property of the workstation rather than of the wording under test.

Neither half of "except 21" is new coverage.
`tests/test_train_classifier.py::test_a_recognized_mode_reaches_its_own_dispatch_branch`
has been exercising the same ten raises on a thirty-row corpus since 2026-10-01, and mode
21 running to completion with no matrix has been exercised since 2026-06-22 by
`tests/test_train_classifier.py::test_train_models_mode21_smoke`, which trains mode 21 with
no binding matrix and asserts that its model and results files exist. It is pinned here
again, beside the ten raises, so that both halves of "except 21" are checked in one file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.train_classifier as train_classifier

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CLASSIFIER = REPO_ROOT / "src" / "train_classifier.py"
CLI = REPO_ROOT / "src" / "cli.py"
USAGE = REPO_ROOT / "USAGE.md"

REQUIREMENT_PHRASE = "required for every feature mode except 21"
USAGE_ROW = "| `--binding-matrix` | Cond. | Required for every feature mode except 21 |"

MODE_WITHOUT_A_MATRIX = 21
MODES_NEEDING_A_MATRIX = tuple(
    mode for mode in train_classifier.RECOGNIZED_FEATURE_MODES if mode != MODE_WITHOUT_A_MATRIX
)

PEPTIDES = (
    "SLYNTVATL",
    "GILGFVFTL",
    "NLVPMVATV",
    "KAFSPEVIP",
    "RAKFKQLLQ",
    "YLLEMLWRL",
    "CLGGLLTMV",
    "TYGPVFMCL",
    "LLWTLVVLL",
    "FLYALALLL",
    "AVFDRKSDA",
    "IVTDFSVIK",
)


def _corpus(tmp_path: Path) -> str:
    """Twelve balanced rows, shaped like tests/test_train_classifier.py's own helper.

    Two of the twelve peptides, CLGGLLTMV and IVTDFSVIK, are in GOLD_STANDARD_EPITOPES, so
    `train_models` holds them out and mode 21 trains on the other ten, still five per label.
    Only peptide and label are needed to reach the dispatch: dropping either raises
    `KeyError`, while dropping virus still runs mode 21 to completion and writes the same
    artifacts, because the CV splitter stratifies on label, origin, supertype and length
    rather than on virus. The virus column is kept because that helper carries one. The
    two `train_models` cases below run with cwd set to their tmp_path, so no proteome
    FASTA is found and every peptide takes the SYNTH_ parent fallback, which none of the
    behaviour under test depends on; the three text-pin tests take no tmp_path and never
    chdir.
    """
    frame = pd.DataFrame(
        {
            "peptide": list(PEPTIDES),
            "label": [index % 2 for index in range(len(PEPTIDES))],
            "virus": ["EBV"] * len(PEPTIDES),
        }
    )
    path = tmp_path / "tiny_corpus.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _carrier_lines(path: Path, needle: str) -> list[str]:
    """Every line of `path` holding `needle`, case-insensitively, stripped.

    Case-insensitive because the phrase opens a sentence in `USAGE.md`'s table cell and
    sits mid-string in the three help strings, so only its first letter differs.
    """
    lowered = needle.lower()
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if lowered in line.lower()
    ]


@pytest.mark.parametrize(
    "mode", MODES_NEEDING_A_MATRIX, ids=[str(mode) for mode in MODES_NEEDING_A_MATRIX]
)
def test_every_recognized_mode_except_21_refuses_to_run_without_a_matrix(
    tmp_path, monkeypatch, capsys, mode
) -> None:
    """Ten modes, ten raises, each naming its own mode: what the wording claims, run."""
    monkeypatch.chdir(tmp_path)
    data_path = _corpus(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        train_classifier.train_models(
            data_path,
            model_dir=str(tmp_path / "models"),
            n_cv_folds=2,
            feature_mode=mode,
            binding_matrix_path=None,
        )
    capsys.readouterr()

    assert str(excinfo.value) == f"--binding-matrix is required for feature-mode {mode}"


def test_mode_21_trains_with_no_binding_matrix(tmp_path, monkeypatch, capsys) -> None:
    """The exception the wording names: 21 is the dispatch's final else and needs no matrix.

    Asserted by running it, because "every mode except 21" is false if 21 also refuses. Its
    artifacts are checked rather than only the absence of a raise, so a future guard that
    stopped a matrix-less mode-21 run earlier could not pass this silently.
    """
    monkeypatch.chdir(tmp_path)
    data_path = _corpus(tmp_path)
    model_dir = tmp_path / "models21"

    train_classifier.train_models(
        data_path,
        model_dir=str(model_dir),
        n_cv_folds=2,
        feature_mode=MODE_WITHOUT_A_MATRIX,
        binding_matrix_path=None,
    )
    capsys.readouterr()

    assert (model_dir / "rf_21feature_legacy.joblib").is_file()
    assert (model_dir / "training_results_mode21.csv").is_file()


def test_the_train_classifier_help_names_every_mode_except_21() -> None:
    """The flag's own help, read as text: the parser is only built under __main__.

    The needle is the help string's opening rather than the filename alone, because two
    other lines of the module name peptide_binding_matrix.csv in prose.
    """
    help_lines = _carrier_lines(TRAIN_CLASSIFIER, 'help="Path to peptide_binding_matrix.csv')

    assert help_lines == [f'help="Path to peptide_binding_matrix.csv ({REQUIREMENT_PHRASE})",']


def test_the_usage_validate_table_names_every_mode_except_21() -> None:
    """USAGE.md's `sestrav validate` row, which named only modes 30 and 31."""
    rows = _carrier_lines(USAGE, "| `--binding-matrix`")

    assert rows == [USAGE_ROW]


def test_all_four_binding_matrix_carriers_hold_one_phrase() -> None:
    """One phrase across the four carriers, so a reader meets the same rule everywhere.

    `src/cli.py`'s two `validate` help strings are read here but not owned here:
    tests/test_cli_validate_default_source.py pins the first by equality on `action.help`
    off `cli._build_validate_parser()`, and the second as a substring of the
    whitespace-collapsed rendering of `cli.main(["validate", "--help"])`. This test is for
    the agreement itself, which no single carrier's own test can see. Each file's matching
    lines are compared as a whole list, so a further line holding this exact phrase, in any
    letter case, in `src/cli.py`, `src/train_classifier.py` or `USAGE.md` also fails it,
    which is the intended review prompt. A carrier worded differently is invisible to it:
    `src/cli.py`'s `validate` runtime note states the same rule as "every feature mode
    except 21 needs one" and is not read here.
    """
    assert _carrier_lines(CLI, REQUIREMENT_PHRASE) == [
        f'help="Path to peptide binding matrix CSV ({REQUIREMENT_PHRASE})",',
        f'help="Binding matrix CSV ({REQUIREMENT_PHRASE})",',
    ]
    assert _carrier_lines(TRAIN_CLASSIFIER, REQUIREMENT_PHRASE) == [
        f'help="Path to peptide_binding_matrix.csv ({REQUIREMENT_PHRASE})",'
    ]
    assert _carrier_lines(USAGE, REQUIREMENT_PHRASE) == [USAGE_ROW]
