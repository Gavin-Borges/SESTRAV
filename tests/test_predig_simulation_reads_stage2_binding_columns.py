"""The simulated PredIG score follows stage 2's binding columns (PC-PG2).

Why this test exists
--------------------
`rule run_predig` (standardize_outputs.smk) hands stage 2's
`results/{proteome_id}_binding.csv` to scripts/run_predig_wrapper.py. With Docker
absent the wrapper writes seeded noise around `bind_val * 0.75`, where `bind_val`
is read from a binding column it matches to the requested allele. It matched a
column only when that column was named exactly like the allele or began with
`HLA-`, while functions/stage2_mhc_binding_prediction.py writes WIDE columns named
by its `_allele_to_col` rule (`'HLA-A*02:01'` -> `'bind_A0201'`) holding
`presentation_score`. No stage-2 column ever matched, so `bind_val` stayed at its
0.5 default on every row and the simulated score was noise around the constant
0.375, carrying no binding signal at all. The wrapper's own comment recorded this
as defect (2).

Pinned here: the wrapper resolves each requested allele to stage 2's `bind_*`
column, so simulated scores track `presentation_score`; a literal `HLA-A*02:01`
column still wins where one exists; an allele with no matching column still falls
back to 0.5; a blank cell falls back to 0.5 rather than clipping to a false 1.0;
the recombinant input PredIG would receive still carries `HLA-...`-style
`HLA_allele` values; and the wrapper's COPY of stage 2's naming rule agrees with
stage 2's own.

Docker is mocked throughout. Nothing here runs PredIG, Docker or the pipeline.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
A0201 = "HLA-A*02:01"
B0702 = "HLA-B*07:02"
PROTEOME_ID = "EBV_B95_8_panel8"

# Eight real 9-mers, so the correlation below is over a spread of presentation
# scores rather than two points. Descending presentation, so a positive
# correlation means the simulated score tracks the binding column and not the row
# order of the CSV.
PEPTIDES = [
    "GILGFVFTL",
    "SLYNTVATL",
    "NLVPMVATV",
    "GLCTLVAML",
    "YVLDHLIVV",
    "KLVALGINAV",
    "AAGIGILTV",
    "ILKEPVHGV",
]
PRESENTATION = [0.98, 0.86, 0.71, 0.58, 0.44, 0.29, 0.13, 0.02]

# bind_val * 0.75 is the wrapper's own transform; 0.5 is its default, so a row that
# matched no column scores noise around this constant.
DEFAULT_CENTRE = 0.5 * 0.75


def _load_wrapper():
    """Load the wrapper by path, the way `rule run_predig` runs it."""
    spec = importlib.util.spec_from_file_location(
        "run_predig_wrapper_simulation_under_test",
        REPO_ROOT / "scripts" / "run_predig_wrapper.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage2_binding(tmp_path: Path, **overrides) -> Path:
    """The shape functions/stage2_mhc_binding_prediction.py writes.

    Best-allele columns, an optional protein_id, and WIDE `bind_*` columns named by
    that module's `_allele_to_col` holding presentation_score. No `HLA-`-named
    column and no column named exactly like an allele, which is the whole defect.
    """
    frame = {
        "peptide": PEPTIDES,
        "allele": [A0201] * len(PEPTIDES),
        "affinity": [12.0] * len(PEPTIDES),
        "presentation_score": PRESENTATION,
        "presentation_percentile": [0.1] * len(PEPTIDES),
        "protein_id": ["P00001"] * len(PEPTIDES),
        "bind_A0201": PRESENTATION,
    }
    frame.update(overrides)
    path = tmp_path / f"{PROTEOME_ID}_binding.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def _simulate(tmp_path, monkeypatch, binding: Path, alleles: str) -> pd.DataFrame:
    """Run the wrapper with Docker unavailable and read the simulated output.

    A missing `docker` binary is FileNotFoundError, an OSError, which is exactly
    what the wrapper's detection treats as "Docker is not usable here".
    """
    wrapper = _load_wrapper()
    monkeypatch.chdir(tmp_path)  # the wrapper writes its temp files under ./results

    def no_docker(*args, **kwargs):
        raise OSError("docker is not installed (test)")

    monkeypatch.setattr(wrapper.subprocess, "run", no_docker)
    out = tmp_path / "results" / f"{PROTEOME_ID}_predig_output.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_predig_wrapper.py",
            "--binding-csv", str(binding),
            "--output", str(out),
            "--alleles", alleles,
            "--proteome-id", PROTEOME_ID,
        ],
    )
    wrapper.main()
    rows = pd.read_csv(out)
    # The simulated flag is unchanged by PC-PG2: these rows are still simulated,
    # they are merely binding-derived now instead of noise around a constant.
    assert (rows["simulated"] == 1).all()
    return rows


# ---------------------------------------------------------------------------
# 1. The simulated score tracks stage 2's wide bind_* column
# ---------------------------------------------------------------------------


def test_simulated_scores_track_the_stage2_presentation_scores(tmp_path, monkeypatch):
    """Over eight peptides the simulated score correlates with presentation_score.

    Before the fix every row drew from the same 0.5 default, so the only variation
    left was the seeded noise and the correlation was whatever that noise happened
    to give. The wrapper seeds with 42, so both the old and the new value of this
    correlation are deterministic.
    """
    rows = _simulate(tmp_path, monkeypatch, _stage2_binding(tmp_path), A0201)
    assert len(rows) == len(PEPTIDES)
    merged = rows.merge(
        pd.DataFrame({"peptide": PEPTIDES, "presentation": PRESENTATION}), on="peptide"
    )
    assert len(merged) == len(PEPTIDES)
    correlation = merged["predig_score"].corr(merged["presentation"])
    assert correlation > 0.9, f"simulated scores do not follow bind_A0201 (r={correlation})"


def test_a_strong_and_a_weak_binder_are_clearly_ordered(tmp_path, monkeypatch):
    """Two peptides at 0.95 and 0.05 presentation, seed-fixed.

    The fixed wrapper centres them on 0.7125 and 0.0375, a gap of 0.675. Before the
    fix both centred on 0.375 and the whole gap was seeded noise, which at this
    seed left them only about 0.08 apart, in either order. The 0.45 threshold sits
    between the two by a wide margin and needs no tolerance tuning.
    """
    binding = _stage2_binding(
        tmp_path,
        peptide=PEPTIDES[:2],
        allele=[A0201] * 2,
        affinity=[12.0, 12.0],
        presentation_score=[0.95, 0.05],
        presentation_percentile=[0.1, 9.0],
        protein_id=["P00001"] * 2,
        bind_A0201=[0.95, 0.05],
    )
    rows = _simulate(tmp_path, monkeypatch, binding, A0201).set_index("peptide")
    strong = rows.loc[PEPTIDES[0], "predig_score"]
    weak = rows.loc[PEPTIDES[1], "predig_score"]
    assert strong - weak > 0.45, f"strong={strong} weak={weak} are not binding-ordered"


# ---------------------------------------------------------------------------
# 2. A literal HLA-named column still wins
# ---------------------------------------------------------------------------


def test_a_literal_hla_allele_column_is_still_honoured(tmp_path, monkeypatch):
    """Some callers pass a frame whose columns ARE the allele names.

    That path worked before PC-PG2 and must keep working. The `bind_A0201` column
    here carries the REVERSED values, so reading the wrong one of the two inverts
    the ordering instead of merely weakening it.
    """
    path = tmp_path / f"{PROTEOME_ID}_binding.csv"
    pd.DataFrame(
        {
            "peptide": PEPTIDES[:2],
            A0201: [0.95, 0.05],
            "bind_A0201": [0.05, 0.95],
        }
    ).to_csv(path, index=False)
    rows = _simulate(tmp_path, monkeypatch, path, A0201).set_index("peptide")
    strong = rows.loc[PEPTIDES[0], "predig_score"]
    weak = rows.loc[PEPTIDES[1], "predig_score"]
    assert strong - weak > 0.45, f"the literal {A0201} column was not used (got {strong}, {weak})"


def test_an_affinity_scale_column_is_still_converted_from_nm(tmp_path, monkeypatch):
    """A value above 1.0 is treated as an IC50 in nM, not as a presentation score.

    The wrapper's own transform, unchanged by PC-PG2: `1 - min(v, 5000) / 5000`. A
    tight binder at 10 nM becomes 0.998 and a non-binder at 5000 nM becomes 0.0,
    so the two stay ordered. Without that branch both would clip to 1.0 and the
    gap would collapse.
    """
    path = tmp_path / f"{PROTEOME_ID}_binding.csv"
    pd.DataFrame({"peptide": PEPTIDES[:2], "bind_A0201": [10.0, 5000.0]}).to_csv(
        path, index=False
    )
    rows = _simulate(tmp_path, monkeypatch, path, A0201).set_index("peptide")
    strong = rows.loc[PEPTIDES[0], "predig_score"]
    weak = rows.loc[PEPTIDES[1], "predig_score"]
    assert strong - weak > 0.45, f"nM affinities were not converted (got {strong}, {weak})"


# ---------------------------------------------------------------------------
# 3. No matching column, and a blank cell, keep the 0.5 default
# ---------------------------------------------------------------------------


def test_an_allele_with_no_binding_column_keeps_the_default(tmp_path, monkeypatch):
    """HLA-B*07:02 is requested but stage 2 wrote no bind_B0702 column.

    Those rows must stay noise around 0.375 rather than borrowing another allele's
    column. The A0201 values are extreme, so borrowing would spread the B0702 rows
    far wider than the noise alone can.
    """
    binding = _stage2_binding(
        tmp_path,
        peptide=PEPTIDES[:2],
        allele=[A0201] * 2,
        affinity=[12.0, 12.0],
        presentation_score=[0.99, 0.01],
        presentation_percentile=[0.1, 9.0],
        protein_id=["P00001"] * 2,
        bind_A0201=[0.99, 0.01],
    )
    rows = _simulate(tmp_path, monkeypatch, binding, f"{A0201},{B0702}")
    assert sorted(rows["allele"].unique()) == sorted([A0201, B0702])
    b_rows = rows[rows["allele"] == B0702]["predig_score"]
    assert len(b_rows) == 2
    # Noise is N(0, 0.12), so a 0.36 half-width is three sigma on a single draw
    # and the seed is fixed; a borrowed bind_A0201 would centre one of these on
    # 0.7425 and the other on 0.0075, both outside.
    assert (b_rows - DEFAULT_CENTRE).abs().max() < 0.36, f"B0702 rows are not at the default: {list(b_rows)}"
    a_rows = rows[rows["allele"] == A0201]["predig_score"]
    assert a_rows.max() - a_rows.min() > 0.45, "the A0201 rows should still be binding-derived"


def test_a_blank_binding_cell_keeps_the_default(tmp_path, monkeypatch):
    """A missing presentation value must not clip to a false MAXIMUM.

    Reachable only now that a stage-2 column is read at all. `min(1.0, nan)`
    returns 1.0, so an unguarded NaN would score the row as the strongest possible
    binder. It must fall back to the 0.5 default instead.
    """
    path = tmp_path / f"{PROTEOME_ID}_binding.csv"
    pd.DataFrame({"peptide": PEPTIDES[:2], "bind_A0201": [None, 0.02]}).to_csv(
        path, index=False
    )
    rows = _simulate(tmp_path, monkeypatch, path, A0201).set_index("peptide")
    blank = rows.loc[PEPTIDES[0], "predig_score"]
    assert abs(blank - DEFAULT_CENTRE) < 0.36, f"a blank cell did not fall back to 0.5: {blank}"
    # 0.75 is bind_val * 0.75 at bind_val == 1.0, the value an unguarded NaN clips
    # to before noise is added.
    assert blank < 0.75 - 0.12, f"a blank cell was read as maximum binding: {blank}"


# ---------------------------------------------------------------------------
# 4. The real-PredIG input's allele names are unchanged
# ---------------------------------------------------------------------------


def test_the_recombinant_input_keeps_hla_style_allele_names(tmp_path, monkeypatch):
    """PC-PG2 must not leak stage 2's column names into what PredIG receives.

    The pairs table feeds `build_recombinant_table`, whose `HLA_allele` column is
    PredIG's input. Mapping alleles to `bind_*` names for the SIMULATED branch must
    leave that as `HLA-A*02:01`, never `bind_A0201`.
    """
    wrapper = _load_wrapper()
    monkeypatch.chdir(tmp_path)
    seen: dict[str, pd.DataFrame] = {}

    def fake_run(cmd, *args, **kwargs):
        if list(cmd[:2]) == ["docker", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=b"Docker version test", stderr=b"")
        if list(cmd[:2]) == ["docker", "run"]:
            # Read inside the fake: the wrapper deletes its temp files afterwards.
            seen["input"] = pd.read_csv(tmp_path / "results" / "temp_predig_input.csv")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    out = tmp_path / "results" / f"{PROTEOME_ID}_predig_output.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_predig_wrapper.py",
            "--binding-csv", str(_stage2_binding(tmp_path)),
            "--output", str(out),
            "--alleles", f"{A0201},{B0702}",
            "--proteome-id", PROTEOME_ID,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        wrapper.main()
    assert exit_info.value.code == 0
    built = seen.get("input")
    assert built is not None, "the PredIG docker command was never invoked"
    assert sorted(built["HLA_allele"].unique()) == sorted([A0201, B0702])
    assert not built["HLA_allele"].astype(str).str.startswith("bind_").any()
    assert sorted(built["epitope"].unique()) == sorted(PEPTIDES)


# ---------------------------------------------------------------------------
# 5. The copied naming rule agrees with stage 2's own
# ---------------------------------------------------------------------------


def test_the_copied_naming_rule_agrees_with_stage2():
    """The wrapper COPIES stage 2's `_allele_to_col` rather than importing it.

    functions/stage2_mhc_binding_prediction.py imports mhcflurry at module level
    (measured in the sestrav env: 3.9 s and over 2,300 modules, torch among them).
    The copy exists to avoid paying that on every wrapper run, and this test is the
    pin that keeps the copy honest.

    This paragraph used to add that environments/predig.yaml "the environment
    `rule run_predig` runs this wrapper under" installs only pandas, numpy and
    docker-py, so importing stage 2 there would raise ModuleNotFoundError. **That
    reason is retracted, and it is the same claim the v2 rebuild of the preceding
    unit was made to retract.** `rule run_predig` DECLARES that conda env
    (standardize_outputs.smk, its `conda:` key) but its shell command invokes the
    wrapper with `{sys.executable}`, Snakemake's own interpreter, and no tracked
    file passes `--use-conda`, so Snakemake never creates or activates it. The
    wrapper therefore runs under whatever interpreter Snakemake is running under,
    which in this repo is the sestrav env, where mhcflurry IS importable. The
    ModuleNotFoundError could not arise, so cost is the whole justification and
    unavailability was never part of it.

    Skipped where mhcflurry is absent, which is a fact about the environment and
    not about the rule: the import was measured working in this repo's sestrav env
    on 2026-10-18, so this does not silently never run there.
    """
    stage2 = pytest.importorskip(
        "functions.stage2_mhc_binding_prediction",
        reason="mhcflurry is not installed in this environment",
    )
    wrapper = _load_wrapper()
    alleles = [
        A0201,
        B0702,
        "HLA-A*01:01",
        "HLA-B*44:02",
        "HLA-C*07:01",
        "HLA-A*11:01",
        "HLA-B*27:05",
        # Shapes the two rules must agree on even though no panel uses them: an
        # allele already stripped of its HLA- prefix, and a four-field name.
        "A*02:01",
        "HLA-A*02:01:01:01",
    ]
    for allele in alleles:
        assert wrapper._stage2_binding_column(allele) == stage2._allele_to_col(allele), allele
    # Not just agreement: the value stage 2 documents for the panel's first allele.
    assert wrapper._stage2_binding_column(A0201) == "bind_A0201"
    assert wrapper._stage2_binding_column(B0702) == "bind_B0702"
