"""
SESTRAV Consecutive Batch Experiment Runner
Orchestrates multiple pipeline and model evaluation trials locally.
Logs results to JSON trial records and maintains an aggregated leaderboard CSV.

Provenance rules enforced by this script
----------------------------------------
* ``--dry-run`` performs NO writes under the output directory. It only prints the
  actions a live run would take. ``results/`` is the provenance root the project's
  certified claims bind to; simulated numbers must never be recorded there.
* Every field written to a trial record or the leaderboard must reflect something
  that actually influenced the run. ``--models``/``--modes`` select the trained
  artifact that Stage 4 scores with (see ``resolve_model_artifact``); an
  unavailable combination fails fast rather than being recorded as if it applied.

Reproducibility caveat
----------------------
``PYTHONHASHSEED`` cannot be pinned from inside a running interpreter - CPython
reads it only at startup. This script therefore seeds the RNGs it can actually
control (``random``, ``numpy``, and ``torch`` when importable) and warns when
``PYTHONHASHSEED`` was not exported by the parent shell. To pin hash
randomisation, export it before launching, e.g.::

    PYTHONHASHSEED=42 python scripts/batch_experiment_runner.py --seeds 42
"""

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("sestrav.batch_runner")

# Feature modes offered by this runner. Every entry has a real dispatch branch in
# src/train_classifier.py. Naming aliases in src/naming.py::MODEL_NAME_ALIASES only cover
# modes that inherited a pre-refactor filename (21/30/31); their absence is not a defect.
#   10  -> binding-only ablation: the 10 per-allele binding scores with no
#          physicochemistry, i.e. the exact converse of 21
#   21  -> sequence-only legacy            (models/<backend>_21feature_legacy.*)
#   30  -> multi-allele legacy             (models/<backend>_30feature_integrated.*)
#   31  -> canonical production (config.yaml feature_mode: 31)
#   33  -> extended (+ netchop/tap; MOCK scores, not real NetChop/TAPreg - D18)
#   35  -> tolerance-aware (+ self-similarity)
#   50  -> expanded multi-allele
# Modes 51/166 and the string modes "30_esm"/"30_graph" exist in train_classifier.py but
# are deliberately not offered here.
#
# NOTE on trained artifacts, corrected 2026-08-17: an earlier version of this comment
# justified excluding 51/166 on the grounds that they have "no released scoring artifact",
# which does not distinguish them from the modes listed above - `git ls-files models/`
# returns ZERO .joblib files, so on a clean clone resolve_model_artifact raises
# FileNotFoundError for EVERY mode here until the user trains one. Mode 10 in particular
# is an evaluation arm with no shipped artifact and no production role. Availability of a
# checkpoint is not the membership criterion; having a dispatch branch is.
VALID_FEATURE_MODES = [10, 21, 30, 31, 33, 35, 50]

# Backends reachable through pipeline.run_pipeline. Selection happens by pointing
# SestravConfig.model_path at the corresponding trained artifact; Stage 4 dispatches
# on the file suffix (.joblib -> sklearn/xgboost, .pt -> PyTorch ANN).
# "gnn" is intentionally absent: the GNN track is a bare state_dict under models/gnn/
# (see src/verify/promote_gnn.py) which ModelRegistry.resolve_model cannot address
# (it resolves basenames directly under models/) and which Stage 4's checkpoint loader
# cannot consume. Offering it would produce a leaderboard row attributed to a model
# that never ran.
VALID_MODEL_BACKENDS = ["rf", "xgb", "ann"]

_BACKEND_SUFFIX = {"rf": ".joblib", "xgb": ".joblib", "ann": ".pt"}


def resolve_model_artifact(model: str, mode: int) -> Path:
    """Map a (backend, feature-mode) pair to its trained artifact under models/.

    Raises FileNotFoundError when no such artifact exists, so a trial can never be
    recorded against a backend/mode combination that did not actually score it.
    (Stage 4 silently falls back to an inline prototype classifier when the model
    file is missing - see functions/stage4_immunogenicity_scoring.py - which would
    otherwise fabricate provenance.)
    """
    stem = f"{model}_{mode}feature_" + ("legacy" if mode == 21 else "integrated")
    artifact = Path("models") / (stem + _BACKEND_SUFFIX[model])
    if not artifact.is_file():
        raise FileNotFoundError(
            f"No trained artifact for backend '{model}' at feature mode {mode}: "
            f"expected {artifact.as_posix()}. Train it first or pick another "
            f"--models/--modes combination."
        )
    return artifact


def seed_everything(seed: int) -> None:
    """Seed the RNGs that can actually be pinned from inside a running interpreter.

    PYTHONHASHSEED is deliberately NOT set here: CPython reads it only at interpreter
    startup, so assigning it to os.environ mid-run is a no-op and would be a
    misleading reproducibility claim. Export it in the parent shell instead.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        logger.warning("numpy unavailable; numpy RNG not seeded")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch unavailable; torch RNG not seeded")


def warn_if_hashseed_unpinned() -> None:
    if os.environ.get("PYTHONHASHSEED") is None:
        logger.warning(
            "PYTHONHASHSEED is not set in the environment. It cannot be pinned from "
            "inside a running interpreter, so hash-order reproducibility is NOT "
            "guaranteed for this batch. Re-launch with PYTHONHASHSEED=<seed> exported "
            "if bit-for-bit reproducibility is required."
        )


def ensure_trials_dir(results_dir: Path) -> Path:
    trials_dir = results_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    return trials_dir


def update_leaderboard(trials_dir: Path, trial_data: dict) -> Path:
    leaderboard_file = trials_dir / "leaderboard.csv"

    columns = [
        "trial_id",
        "timestamp",
        "feature_mode",
        "model_type",
        "model_artifact",
        "seed",
        "proteome_id",
        "total_peptides",
        "mean_score",
        "top5_avg_score",
        "duration_sec",
        "status",
        "notes",
    ]

    row = {
        "trial_id": trial_data.get("trial_id"),
        "timestamp": trial_data.get("timestamp"),
        "feature_mode": trial_data.get("feature_mode"),
        "model_type": trial_data.get("model_type"),
        "model_artifact": trial_data.get("model_artifact", ""),
        "seed": trial_data.get("seed"),
        "proteome_id": trial_data.get("proteome_id"),
        "total_peptides": trial_data.get("total_peptides", 0),
        "mean_score": trial_data.get("mean_score", 0.0),
        "top5_avg_score": trial_data.get("top5_avg_score", 0.0),
        "duration_sec": trial_data.get("duration_sec", 0.0),
        "status": trial_data.get("status", "UNKNOWN"),
        "notes": trial_data.get("notes", ""),
    }

    df_row = pd.DataFrame([row], columns=columns)

    if leaderboard_file.exists():
        # Refuse to append under a mismatched header: silently writing rows whose
        # columns do not line up with the existing header corrupts the ledger.
        with open(leaderboard_file, "r", encoding="utf-8") as f:
            existing_header = f.readline().strip()
        if existing_header and existing_header != ",".join(columns):
            raise ValueError(
                f"Leaderboard schema mismatch in {leaderboard_file}: existing header "
                f"'{existing_header}' does not match the current columns. Archive or "
                f"migrate the old file before appending."
            )
        df_row.to_csv(leaderboard_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(leaderboard_file, mode="w", header=True, index=False)

    return leaderboard_file


def run_single_trial(
    trial_id: str,
    mode: int,
    model: str,
    seed: int,
    antigen: str,
    results_dir: Path,
    dry_run: bool = False,
) -> dict:
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        f"--- Starting Trial {trial_id} | Mode: {mode} | Model: {model} | Seed: {seed} | Antigen: {antigen} ---"
    )

    if dry_run:
        # A dry run must not touch the output directory at all: results/ is the
        # provenance root the project's certified claims bind to. Report the planned
        # actions and return a record that carries no simulated metrics.
        trials_dir = results_dir / "trials"
        try:
            artifact = resolve_model_artifact(model, mode).as_posix()
            artifact_note = f"would score with {artifact}"
        except FileNotFoundError as e:
            artifact = ""
            artifact_note = f"model artifact MISSING -> live run would abort ({e})"

        logger.info("[DRY RUN] No files will be written. Planned actions:")
        logger.info(f"[DRY RUN]   ensure directory        : {trials_dir.as_posix()}")
        logger.info(f"[DRY RUN]   write trial record      : {(trials_dir / f'trial_{trial_id}.json').as_posix()}")
        logger.info(f"[DRY RUN]   append leaderboard row  : {(trials_dir / 'leaderboard.csv').as_posix()}")
        logger.info(f"[DRY RUN]   run_pipeline(antigen={antigen!r}, feature_mode={mode}, seed={seed})")
        logger.info(f"[DRY RUN]   {artifact_note}")

        return {
            "trial_id": trial_id,
            "timestamp": timestamp,
            "feature_mode": mode,
            "model_type": model,
            "model_artifact": artifact,
            "seed": seed,
            "proteome_id": antigen,
            # No metrics: nothing was computed, so nothing is reported.
            "total_peptides": None,
            "mean_score": None,
            "top5_avg_score": None,
            "duration_sec": None,
            "status": "PLANNED (DRY_RUN, not executed, nothing written)",
            "notes": artifact_note,
        }

    model_artifact = ""
    try:
        # Import pipeline functions internally to avoid loading before args check
        from pipeline import run_pipeline
        from src.core.config import SestravConfig
        from src.core.model_registry import ModelRegistry

        cfg = SestravConfig.load()
        cfg.output_dir = results_dir
        cfg.mc_dropout = True

        # Feature mode and backend are applied by selecting the trained artifact that
        # Stage 4 scores with; run_pipeline resolves cfg.model_path through
        # ModelRegistry. This is the only hook the pipeline exposes for model choice.
        artifact_path = resolve_model_artifact(model, mode)
        cfg.model_path = artifact_path
        model_artifact = artifact_path.as_posix()
        # Keep the config's declared feature_mode consistent with the artifact actually
        # loaded, so anything downstream that reads it cannot disagree with reality.
        cfg.feature_mode = mode
        logger.info(f"Trial {trial_id}: scoring with {model_artifact} (feature_mode={mode})")

        seed_everything(seed)

        registry = ModelRegistry(cfg)
        fasta_path = os.path.join("data", "proteomes", f"{antigen}.fasta")

        if not os.path.exists(fasta_path):
            # Fallback to direct proteome resolution if fasta is not in default folder
            fasta_path = cfg.proteome_files.get(antigen, fasta_path)

        run_pipeline(antigen, fasta_path, cfg, registry)

        duration = round(time.time() - start_time, 3)

        # Inspect output ranked.csv if generated. Stage 4 writes to a literal
        # "results/" directory regardless of cfg.output_dir, so check both.
        ranked_csv = results_dir / f"{antigen}_ranked.csv"
        if not ranked_csv.exists():
            ranked_csv = Path("results") / f"{antigen}_ranked.csv"
        total_peptides = 0
        mean_score = 0.0
        top5_avg = 0.0

        if ranked_csv.exists():
            df_ranked = pd.read_csv(ranked_csv)
            total_peptides = len(df_ranked)
            if "immunogenicity_score" in df_ranked.columns:
                mean_score = round(float(df_ranked["immunogenicity_score"].mean()), 4)
                top5_avg = round(float(df_ranked.head(5)["immunogenicity_score"].mean()), 4)
            status = "SUCCESS"
            notes = f"Ranked file generated at {ranked_csv.as_posix()}"
        else:
            status = "NO_OUTPUT"
            notes = f"run_pipeline completed but no ranked CSV found for {antigen}"

        trial_record = {
            "trial_id": trial_id,
            "timestamp": timestamp,
            "feature_mode": mode,
            "model_type": model,
            "model_artifact": model_artifact,
            "seed": seed,
            "proteome_id": antigen,
            "total_peptides": total_peptides,
            "mean_score": mean_score,
            "top5_avg_score": top5_avg,
            "duration_sec": duration,
            "status": status,
            "notes": notes,
        }
    except Exception as e:
        duration = round(time.time() - start_time, 3)
        logger.error(f"Trial {trial_id} failed: {e}", exc_info=True)
        trial_record = {
            "trial_id": trial_id,
            "timestamp": timestamp,
            "feature_mode": mode,
            "model_type": model,
            "model_artifact": model_artifact,
            "seed": seed,
            "proteome_id": antigen,
            "total_peptides": 0,
            "mean_score": 0.0,
            "top5_avg_score": 0.0,
            "duration_sec": duration,
            "status": f"FAILED: {type(e).__name__}",
            "notes": str(e),
        }

    trials_dir = ensure_trials_dir(results_dir)
    json_path = trials_dir / f"trial_{trial_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trial_record, f, indent=2)

    update_leaderboard(trials_dir, trial_record)
    logger.info(f"Trial {trial_id} complete. Duration: {trial_record['duration_sec']}s | Status: {trial_record['status']}")

    return trial_record


def main():
    parser = argparse.ArgumentParser(description="SESTRAV Batch Experiment Runner")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per parameter set")
    parser.add_argument(
        "--modes",
        nargs="+",
        type=int,
        default=[31],
        choices=VALID_FEATURE_MODES,
        help="Feature modes to evaluate (default 31, the canonical production mode in config.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["rf"],
        choices=VALID_MODEL_BACKENDS,
        help=(
            "Model backends. Selects the trained artifact Stage 4 scores with. "
            "'gnn' is not offered: the GNN track is not reachable through run_pipeline."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    parser.add_argument(
        "--antigens",
        nargs="+",
        type=str,
        default=["EBV_B95_8_panel8"],
        help="Antigens/Proteome IDs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate trial execution without heavy computation")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to write results and trials")

    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    trials_dir = results_dir / "trials"

    if args.dry_run:
        # No mkdir, no trial JSON, no leaderboard append: a dry run leaves the
        # provenance root byte-for-byte untouched.
        logger.info("DRY RUN: no directories will be created and no files written.")
    else:
        # Pre-flight: every requested (backend, mode) pair must map to a real artifact
        # before anything is written, so a batch cannot half-populate the leaderboard.
        missing = []
        for mode in args.modes:
            for model in args.models:
                try:
                    resolve_model_artifact(model, mode)
                except FileNotFoundError as e:
                    missing.append(str(e))
        if missing:
            for msg in missing:
                logger.error(msg)
            sys.exit(2)

        results_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = ensure_trials_dir(results_dir)
        warn_if_hashseed_unpinned()

    all_records = []
    run_counter = 1

    logger.info("==================================================")
    logger.info(f" Starting SESTRAV Batch Runner ({'DRY RUN' if args.dry_run else 'LIVE RUN'})")
    logger.info(f" Modes: {args.modes} | Models: {args.models} | Seeds: {args.seeds}")
    logger.info(f" Antigens: {args.antigens} | Output Dir: {results_dir.resolve()}")
    logger.info("==================================================")

    for mode in args.modes:
        for model in args.models:
            for seed in args.seeds:
                for antigen in args.antigens:
                    for iteration in range(1, args.trials + 1):
                        trial_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_counter:03d}"
                        record = run_single_trial(
                            trial_id=trial_id,
                            mode=mode,
                            model=model,
                            seed=seed,
                            antigen=antigen,
                            results_dir=results_dir,
                            dry_run=args.dry_run,
                        )
                        all_records.append(record)
                        run_counter += 1

    logger.info("\n==================================================")
    if args.dry_run:
        logger.info(f" Batch Runner Finished (DRY RUN). Planned {len(all_records)} trial(s); wrote nothing.")
        logger.info(f" Live run would write: {(trials_dir / 'leaderboard.csv').as_posix()}")
    else:
        logger.info(f" Batch Runner Finished. Completed {len(all_records)} trial(s).")
        logger.info(f" Trial Leaderboard CSV: {(trials_dir / 'leaderboard.csv').resolve()}")
    logger.info("==================================================")

    # Print summary table
    df_summary = pd.DataFrame(all_records)
    print("\n--- BATCH RUN SUMMARY (DRY RUN: nothing written) ---" if args.dry_run else "\n--- BATCH RUN SUMMARY ---")
    print(
        df_summary[
            [
                "trial_id",
                "feature_mode",
                "model_type",
                "model_artifact",
                "seed",
                "mean_score",
                "top5_avg_score",
                "duration_sec",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
