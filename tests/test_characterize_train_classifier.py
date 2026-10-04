"""Characterisation tests for src/train_classifier.py.

src/train_classifier.py carries a __main__ guard, so .coveragerc.library omits
it and the required coverage gate never measures it. Before this file,
train_models had only ever trained feature modes 10, 21 and 31 under the suite,
had never mapped a single peptide to a real parent protein, and the module
entry point's input-validation errors after the data-file check had never
fired.

Every test here except test_train_models_names_the_unknown_mode_it_rejects PINS
CURRENT BEHAVIOUR; that one asserts the guard's refusal and is counter-marked.
None of the others asserts that the behaviour is right; several record
behaviour an owner may want to change, and say so in their docstrings. Nothing
in src/train_classifier.py is edited. All inputs are synthetic frames written
to tmp_path, or the tracked proteome FASTAs; no network, no ESM weights (the
ESM call is replaced by a deterministic stub), and no model or dataset artifact
from the working tree.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.train_classifier as tc
from src.features import (
    BINDING_ALLELE_COLUMNS,
    FEATURE_COLUMNS_30,
    FEATURE_COLUMNS_30_ESM,
    FEATURE_COLUMNS_30_GRAPH,
    FEATURE_COLUMNS_33,
    FEATURE_COLUMNS_35,
    FEATURE_COLUMNS_50,
    FEATURE_COLUMNS_51,
    FEATURE_COLUMNS_ALLELE,
    HLA_PSEUDO_COLS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _peptides(n):
    """Distinct synthetic peptides, alternating 9-mers and 10-mers, standard residues only."""
    out = []
    for i in range(n):
        core = _AAS[i % 20] + _AAS[(i // 20) % 20] + "LMWITQV"
        out.append(core if i % 2 == 0 else core + "L")
    return out


def _corpus(n=24, pseudo=False, virus=True):
    peps = _peptides(n)
    frame = {
        "peptide": peps,
        "label": ([0, 1, 1, 0] * n)[:n],
        "hla_allele": ["HLA-A*02:01" if i % 2 else "HLA-C*99:99" for i in range(n)],
    }
    if virus:
        frame["virus"] = ["HPV16" if i % 3 == 0 else "EBV" for i in range(n)]
    df = pd.DataFrame(frame)
    if pseudo:
        for j, col in enumerate(HLA_PSEUDO_COLS):
            df[col] = [((i + j) % 7) / 7.0 for i in range(n)]
    return df


def _write_inputs(tmp_path, df):
    """Write the corpus plus a binding matrix and both caches; return their paths."""
    peps = list(df["peptide"])
    n = len(peps)
    data = tmp_path / "corpus.csv"
    df.to_csv(data, index=False)

    binding = {"peptide": peps}
    for k, col in enumerate(BINDING_ALLELE_COLUMNS):
        binding[col] = [((i * 3 + k) % 10) / 10.0 for i in range(n)]
    binding_path = tmp_path / "binding.csv"
    pd.DataFrame(binding).to_csv(binding_path, index=False)

    # Only every other peptide is in the antigen-processing cache, so imputation runs.
    ap_path = tmp_path / "ap_cache.csv"
    pd.DataFrame(
        {
            "peptide": peps[::2],
            "netchop_score": [i / n for i in range(0, n, 2)],
            "tap_score": [1 - i / n for i in range(0, n, 2)],
        }
    ).to_csv(ap_path, index=False)

    sim_path = tmp_path / "sim_cache.csv"
    pd.DataFrame(
        {
            "peptide": peps,
            "self_similarity_max_identity": [(i % 5) / 5.0 for i in range(n)],
            "self_similarity_exact_match": [float(i % 5 == 4) for i in range(n)],
        }
    ).to_csv(sim_path, index=False)
    return {
        "data": str(data),
        "binding": str(binding_path),
        "ap": str(ap_path),
        "sim": str(sim_path),
    }


def _fake_esm(peptide):
    return np.full(320, len(peptide) / 10.0)


# ---------------------------------------------------------------------------
# train_models: feature-mode dispatch
# ---------------------------------------------------------------------------

_DISPATCH = [
    (30, FEATURE_COLUMNS_30),
    (33, FEATURE_COLUMNS_33),
    (35, FEATURE_COLUMNS_35),
    (50, FEATURE_COLUMNS_50),
    (51, FEATURE_COLUMNS_51),
    (166, FEATURE_COLUMNS_ALLELE),
    ("30_graph", FEATURE_COLUMNS_30_GRAPH),
    ("30_esm", FEATURE_COLUMNS_30_ESM),
]


@pytest.mark.parametrize("mode, expected", _DISPATCH, ids=[str(m) for m, _ in _DISPATCH])
def test_train_models_dispatch_builds_the_requested_matrix(
    tmp_path, monkeypatch, capsys, mode, expected
):
    # pins behaviour, not correctness
    inputs = _write_inputs(tmp_path, _corpus(pseudo=(mode == 166)))
    if mode == "30_esm":
        monkeypatch.setattr(tc, "get_esm_cls_token", _fake_esm)
    needs_ap = mode in (33, 35)
    model_dir = tmp_path / "m"

    rf, xgb, _rf_avg, _xgb_avg = tc.train_models(
        inputs["data"],
        model_dir=str(model_dir),
        n_cv_folds=2,
        feature_mode=mode,
        binding_matrix_path=inputs["binding"],
        antigen_processing_cache_path=inputs["ap"] if needs_ap else None,
        self_similarity_cache_path=inputs["sim"] if mode == 35 else None,
    )
    capsys.readouterr()

    assert list(rf.feature_names_in_) == list(expected)
    imp = pd.read_csv(model_dir / "feature_importances.csv")
    assert sorted(imp["feature"]) == sorted(expected)
    rf_stem, xgb_stem = tc._artifact_stems(mode)
    assert (model_dir / f"{rf_stem}.joblib").is_file()
    assert (model_dir / f"{xgb_stem}.joblib").is_file()


_REQUIRES = [
    (166, (), "--binding-matrix is required for feature-mode 166"),
    (51, (), "--binding-matrix is required for feature-mode 51"),
    (50, (), "--binding-matrix is required for feature-mode 50"),
    ("30_esm", (), "--binding-matrix is required for feature-mode 30_esm"),
    ("30_graph", (), "--binding-matrix is required for feature-mode 30_graph"),
    (35, (), "--binding-matrix is required for feature-mode 35"),
    (35, ("binding",), "--antigen-processing-cache is required for feature-mode 35"),
    (35, ("binding", "ap"), "--self-similarity-cache is required for feature-mode 35"),
    (33, (), "--binding-matrix is required for feature-mode 33"),
    (33, ("binding",), "--antigen-processing-cache is required for feature-mode 33"),
    (30, (), "--binding-matrix is required for feature-mode 30"),
]


@pytest.mark.parametrize(
    "mode, given, message",
    _REQUIRES,
    ids=[f"{m}-given-{'+'.join(g) or 'none'}" for m, g, _ in _REQUIRES],
)
def test_train_models_missing_input_raises_after_creating_model_dir(
    tmp_path, capsys, mode, given, message
):
    """Each mode's missing-input ValueError fires only after model_dir is created.

    The check runs after the corpus is read and every peptide is mapped, so a
    rejected run still leaves an empty model_dir behind.
    """
    # pins behaviour, not correctness
    inputs = _write_inputs(tmp_path, _corpus(pseudo=(mode == 166)))
    model_dir = tmp_path / "m"
    with pytest.raises(ValueError, match=f"^{message}$"):
        tc.train_models(
            inputs["data"],
            model_dir=str(model_dir),
            n_cv_folds=2,
            feature_mode=mode,
            binding_matrix_path=inputs["binding"] if "binding" in given else None,
            antigen_processing_cache_path=inputs["ap"] if "ap" in given else None,
        )
    capsys.readouterr()
    assert model_dir.is_dir()
    assert list(model_dir.iterdir()) == []


@pytest.mark.parametrize(
    "mode, shown",
    [(99, "99"), ("31x", "'31x'"), ("099", "99")],
    ids=["int-99", "str-31x", "str-099-coerced"],
)
def test_train_models_names_the_unknown_mode_it_rejects(tmp_path, mode, shown):
    """An unrecognised feature_mode is refused, named as coerced, before anything is written.

    This test asserts DESIRED behaviour and is the one exception to this file's
    convention, so it carries the counter-marker below in place of the usual
    `pins behaviour` one. It previously pinned the defect: the dispatch chain's
    final `else` built the 21-feature sequence-only matrix while
    _artifact_stems keyed off the REQUESTED mode, so a mode that reached
    train_models unchecked wrote a 21-feature fit as
    rf_99feature_integrated.joblib and exited 0. Through the CLI it was
    reachable only via `sestrav validate`, from its `--feature-mode` flag,
    which has no `choices` list, or, absent the flag, from config.yaml's
    `feature_mode`; cmd_validate coerces either and forwards whatever it is
    given. A direct call to train_models reached it too. `python -m
    src.train_classifier --feature-mode 99` never could, because that parser's
    flag does carry a `choices` list and argparse exits 2 with "invalid
    choice". The guard at the top of train_models closes all three routes.

    It asserts the EXACT message, which the guard's own test in
    tests/test_train_classifier.py does not: that one passes the substring
    "Unknown feature_mode" to pytest.raises(match=...), which is re.search and
    so matches anywhere in the message, so it still passes if the guard names
    the wrong mode or drops the `!r`. The `!r` is load-bearing, and the first
    two ids are chosen to show it - an int mode is reported bare and a
    non-numeric string is reported quoted, because the int() coercion above the
    guard leaves the original value in place when it fails. The third shows the
    coercion itself: "099" is reported as 99, not as '099', so a guard that
    named the mode as given rather than as coerced fails it. The recognised
    list is read from the module rather than retyped, since a separate test
    ties that tuple to the module CLI's own choices. model_dir is asserted
    ABSENT because the guard runs before os.makedirs, so a refused run writes
    nothing at all, and the inputs are written and valid, so the mode is the
    only thing left to raise about.
    """
    # asserts desired behaviour, unlike every other test in this file
    inputs = _write_inputs(tmp_path, _corpus())
    model_dir = tmp_path / "m"
    expected = f"Unknown feature_mode {shown}; expected one of: " + ", ".join(
        str(recognised) for recognised in tc.RECOGNIZED_FEATURE_MODES
    )

    with pytest.raises(ValueError) as excinfo:
        tc.train_models(
            inputs["data"], model_dir=str(model_dir), n_cv_folds=2, feature_mode=mode
        )

    assert str(excinfo.value) == expected
    assert not model_dir.exists()


# ---------------------------------------------------------------------------
# train_models: what the final refit does and does not see
# ---------------------------------------------------------------------------


def test_final_refit_ignores_sample_weights(tmp_path, capsys):
    """use_sample_weights reaches the CV folds only; the serialized models are unweighted.

    OWNER LOOK: rf_final.fit and xgb_final.fit are called with (X, y) alone, so
    two runs differing only in use_sample_weights ship byte-for-byte the same
    feature importances. A CV figure reported "with sample weights" therefore
    describes a model that is not the artifact written to model_dir.
    """
    # pins behaviour, not correctness
    inputs = _write_inputs(tmp_path, _corpus(n=32))
    importances = {}
    for weighted in (False, True):
        model_dir = tmp_path / f"w{int(weighted)}"
        tc.train_models(
            inputs["data"],
            model_dir=str(model_dir),
            n_cv_folds=2,
            feature_mode=21,
            use_sample_weights=weighted,
        )
        importances[weighted] = pd.read_csv(model_dir / "feature_importances.csv")
    out = capsys.readouterr().out

    assert "Sample weights enabled" in out
    lo, hi = out.split("Weight range: [", 1)[1].split("]", 1)[0].split(", ")
    assert float(lo) < float(hi)
    pd.testing.assert_frame_equal(importances[False], importances[True])


def test_sample_weights_without_a_virus_column_skip_the_rate_line(tmp_path, capsys):
    # pins behaviour, not correctness
    inputs = _write_inputs(tmp_path, _corpus(virus=False))
    tc.train_models(
        inputs["data"],
        model_dir=str(tmp_path / "m"),
        n_cv_folds=2,
        feature_mode=21,
        use_sample_weights=True,
    )
    out = capsys.readouterr().out
    assert "Sample weights enabled" in out
    assert "9-mer fraction: 50.00%" in out
    assert "EBV positive rate" not in out


def test_fold_impute_changes_cv_only_and_ships_the_same_final_model(tmp_path, capsys):
    """With fold_impute the final refit still uses whole-cache medians.

    Pins the docstring's claim that the serialized artifact is unchanged by the
    in-fold repair: feature importances match the fold_impute=False run.
    """
    # pins behaviour, not correctness
    inputs = _write_inputs(tmp_path, _corpus(n=32))
    importances = {}
    for fold_impute in (False, True):
        model_dir = tmp_path / f"f{int(fold_impute)}"
        tc.train_models(
            inputs["data"],
            model_dir=str(model_dir),
            n_cv_folds=2,
            feature_mode=33,
            binding_matrix_path=inputs["binding"],
            antigen_processing_cache_path=inputs["ap"],
            fold_impute=fold_impute,
        )
        importances[fold_impute] = pd.read_csv(model_dir / "feature_importances.csv")
    out = capsys.readouterr().out

    assert "left as NaN for in-fold median imputation" in out
    assert "Imputed 16 missing antigen processing scores with cache medians" in out
    pd.testing.assert_frame_equal(importances[False], importances[True])


def test_programmatic_defaults_are_ungrouped_and_whole_cache():
    """train_models defaults differ from the module entry point's defaults.

    The entry point resolves --cv-group-by to "peptide" and --no-fold-impute to
    fold_impute=True; a direct call (as `sestrav validate` makes) gets None and
    False.
    """
    # pins behaviour, not correctness
    params = inspect.signature(tc.train_models).parameters
    assert params["cv_group_by"].default is None
    assert params["fold_impute"].default is False
    assert params["use_sample_weights"].default is False


# ---------------------------------------------------------------------------
# Parent-protein mapping
# ---------------------------------------------------------------------------


def test_train_models_maps_proteome_substrings_regardless_of_row_virus(
    tmp_path, monkeypatch, capsys
):
    """A peptide found in a tracked proteome FASTA takes the FIRST protein containing it.

    The row's virus label plays no part: a ZIKV-labelled row carrying an EBV
    substring is attributed to the EBV protein.
    """
    # pins behaviour, not correctness
    monkeypatch.chdir(REPO_ROOT)
    proteins = tc.load_all_proteins()
    name0, seq0 = next(iter(proteins.items()))
    real = [seq0[10:19], seq0[30:40]]
    df = _corpus(n=24)
    df.loc[0, "peptide"] = real[0]
    df.loc[1, "peptide"] = real[1]
    df.loc[[0, 1], "virus"] = "ZIKV"
    inputs = _write_inputs(tmp_path, df)

    tc.train_models(inputs["data"], model_dir=str(tmp_path / "m"), n_cv_folds=2, feature_mode=21)
    out = capsys.readouterr().out

    oof = pd.read_csv(tmp_path / "m" / "rf_oof_predictions.csv").set_index("peptide")
    for pep in real:
        expected = next(name for name, seq in proteins.items() if pep in seq)
        assert oof.loc[pep, "protein"] == expected
        assert oof.loc[pep, "virus"] == "ZIKV"
    assert oof.loc[real[0], "protein"] == name0
    others = [p for p in df["peptide"] if p not in real]
    assert all(oof.loc[p, "protein"] == f"SYNTH_{p}" for p in others)
    assert "Mapped 2 peptides to parent proteins. Synthetic fallback used for 22 peptides." in out


def test_load_all_proteins_resolves_paths_against_the_working_directory(tmp_path, monkeypatch):
    """OWNER LOOK: the FASTA list is relative, so any other working directory yields {}.

    train_models then maps every peptide to SYNTH_<peptide>, with no warning.
    """
    # pins behaviour, not correctness
    monkeypatch.chdir(tmp_path)
    assert tc.load_all_proteins() == {}


def test_load_all_proteins_parsing_quirks(tmp_path, monkeypatch):
    """Blank lines are skipped, sequences are upper-cased, an empty file is harmless,
    only sp| headers are split, and a later duplicate name overwrites an earlier one
    while keeping the earlier key position.
    """
    # pins behaviour, not correctness
    monkeypatch.chdir(tmp_path)
    prot = tmp_path / "data" / "proteomes"
    prot.mkdir(parents=True)
    (prot / "EBV_B95_8_panel8.fasta").write_text(
        ">sp|P0|AAA_TEST first\nMKT\n\nvls\n>DUP_TEST panel copy\nAAAA\n", encoding="utf-8"
    )
    (prot / "HPV16_18_panel8.fasta").write_text("", encoding="utf-8")
    (prot / "EBV_B95_8_uniprot_reviewed.fasta").write_text(
        ">DUP_TEST uniprot copy\nCCCC\n>tr|Q1|TRM_TEST x\nwwww\n", encoding="utf-8"
    )

    proteins = tc.load_all_proteins()

    assert proteins == {"AAA_TEST": "MKTVLS", "DUP_TEST": "CCCC", "tr|Q1|TRM_TEST": "WWWW"}
    assert list(proteins) == ["AAA_TEST", "DUP_TEST", "tr|Q1|TRM_TEST"]


def test_protein_name_splits_swissprot_headers_only():
    # pins behaviour, not correctness
    assert tc._get_protein_name_from_header("sp|P03129|VE7_HPV16 Protein E7") == "VE7_HPV16"
    assert tc._get_protein_name_from_header("tr|A0A0B4|X_Y some protein") == "tr|A0A0B4|X_Y"
    assert tc._get_protein_name_from_header("") == ""
    assert tc._get_protein_name_from_header("   ") == ""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_filter_quarantined_is_silent_when_nothing_is_dropped(capsys):
    # pins behaviour, not correctness
    df = pd.DataFrame({"peptide": ["A", "B"], "is_quarantined": [False, False]})
    out_df = tc._filter_quarantined(df)
    assert capsys.readouterr().out == ""
    pd.testing.assert_frame_equal(out_df, df)
    assert out_df is not df


def test_filter_quarantined_treats_any_non_empty_string_as_quarantined(capsys):
    """The flag is cast with astype(bool), so the STRING "False" drops the row.

    A CSV holding True/False parses to real booleans and is unaffected; a frame
    whose flag arrives as text ("False", "no") loses those rows.
    """
    # pins behaviour, not correctness
    df = pd.DataFrame(
        {"peptide": ["A", "B", "C", "D"], "is_quarantined": ["False", "True", "no", ""]}
    )
    out_df = tc._filter_quarantined(df)
    capsys.readouterr()
    assert out_df["peptide"].tolist() == ["D"]


def test_mode_166_artifact_stems():
    # pins behaviour, not correctness
    names = [os.path.basename(p) for p in tc.planned_artifact_paths("d", 166)]
    assert names[:2] == ["rf_166feature_allele_aware.joblib", "xgb_166feature_allele_aware.joblib"]
    assert "training_results_mode166.csv" in names


def test_prepare_features_166_rejects_a_partial_binding_matrix_with_a_shape_error(tmp_path):
    """Five of ten allele columns: modes 10/30/50 raise a named ValueError, 166 does not.

    Mode 166 zero-fills a matrix with NO allele columns but a PARTIAL one
    reaches pandas with 5 values per row against 10 column names and fails
    with pandas' own shape message.
    """
    # pins behaviour, not correctness
    df = _corpus(n=6, pseudo=True)
    matrix = {"peptide": list(df["peptide"])}
    for col in BINDING_ALLELE_COLUMNS[:5]:
        matrix[col] = [0.5] * len(df)
    path = tmp_path / "partial.csv"
    pd.DataFrame(matrix).to_csv(path, index=False)
    with pytest.raises(ValueError, match="10 columns passed, passed data had 5 columns"):
        tc.prepare_features_166(df, str(path))


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

_ENTRY_CASES = [
    ("binding", ["--binding-matrix", "absent_matrix.csv"], 2, "Binding matrix file does not exist"),
    (
        "ap-cache",
        ["--antigen-processing-cache", "absent_ap.csv"],
        2,
        "Antigen processing cache does not exist",
    ),
    (
        "sim-cache",
        ["--self-similarity-cache", "absent_sim.csv"],
        2,
        "Self-similarity cache does not exist",
    ),
    ("guard", [], 1, "FileExistsError"),
]


@pytest.mark.parametrize("case, extra, rc, needle", _ENTRY_CASES, ids=[c[0] for c in _ENTRY_CASES])
def test_entry_point_validates_inputs_after_the_data_file(tmp_path, case, extra, rc, needle):
    # pins behaviour, not correctness
    data = tmp_path / "corpus.csv"
    _corpus(n=8).to_csv(data, index=False)
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "training_results.csv").write_text("metric\n", encoding="utf-8")
    argv = [
        sys.executable,
        "-m",
        "src.train_classifier",
        "--data",
        str(data),
        "--model-dir",
        str(model_dir),
        *[str(tmp_path / a) if a.startswith("absent") else a for a in extra],
    ]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300)
    assert result.returncode == rc
    assert needle in result.stderr
    assert (model_dir / "training_results.csv").read_text(encoding="utf-8") == "metric\n"
