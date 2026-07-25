"""
SESTRAV Consecutive Batch Experiment Runner
Orchestrates multiple pipeline and model evaluation trials locally.
Logs results to JSON trial records and maintains an aggregated leaderboard CSV.
"""

import argparse
import datetime
import json
import logging
import os
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
        time.sleep(0.5)  # Simulate execution
        duration = round(time.time() - start_time, 3)
        trial_record = {
            "trial_id": trial_id,
            "timestamp": timestamp,
            "feature_mode": mode,
            "model_type": model,
            "seed": seed,
            "proteome_id": antigen,
            "total_peptides": 120,
            "mean_score": 0.4521,
            "top5_avg_score": 0.8912,
            "duration_sec": duration,
            "status": "SUCCESS (DRY_RUN)",
            "notes": "Dry run verification trial",
        }
    else:
        try:
            # Import pipeline functions internally to avoid loading before args check
            from pipeline import run_pipeline
            from src.core.config import SestravConfig
            from src.core.model_registry import ModelRegistry

            cfg = SestravConfig.load()
            cfg.output_dir = results_dir
            cfg.mc_dropout = True
            
            # Apply feature mode if supported by config/env
            os.environ["SESTRAV_FEATURE_MODE"] = str(mode)
            os.environ["PYTHONHASHSEED"] = str(seed)

            registry = ModelRegistry(cfg)
            fasta_path = os.path.join("data", "proteomes", f"{antigen}.fasta")

            if not os.path.exists(fasta_path):
                # Fallback to direct proteome resolution if fasta is not in default folder
                fasta_path = cfg.proteome_files.get(antigen, fasta_path)

            run_pipeline(antigen, fasta_path, cfg, registry)

            duration = round(time.time() - start_time, 3)

            # Inspect output ranked.csv if generated
            ranked_csv = results_dir / f"{antigen}_ranked.csv"
            total_peptides = 0
            mean_score = 0.0
            top5_avg = 0.0

            if ranked_csv.exists():
                df_ranked = pd.read_csv(ranked_csv)
                total_peptides = len(df_ranked)
                if "immunogenicity_score" in df_ranked.columns:
                    mean_score = round(float(df_ranked["immunogenicity_score"].mean()), 4)
                    top5_avg = round(float(df_ranked.head(5)["immunogenicity_score"].mean()), 4)

            trial_record = {
                "trial_id": trial_id,
                "timestamp": timestamp,
                "feature_mode": mode,
                "model_type": model,
                "seed": seed,
                "proteome_id": antigen,
                "total_peptides": total_peptides,
                "mean_score": mean_score,
                "top5_avg_score": top5_avg,
                "duration_sec": duration,
                "status": "SUCCESS",
                "notes": f"Ranked file generated at {ranked_csv.name}",
            }
        except Exception as e:
            duration = round(time.time() - start_time, 3)
            logger.error(f"Trial {trial_id} failed: {e}", exc_info=True)
            trial_record = {
                "trial_id": trial_id,
                "timestamp": timestamp,
                "feature_mode": mode,
                "model_type": model,
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
        "--modes", nargs="+", type=int, default=[30], choices=[21, 30, 50], help="Feature modes to evaluate"
    )
    parser.add_argument(
        "--models", nargs="+", type=str, default=["ann"], choices=["ann", "gnn", "rf", "xgb"], help="Model backends"
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
    results_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = ensure_trials_dir(results_dir)

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
    logger.info(f" Batch Runner Finished. Completed {len(all_records)} trial(s).")
    logger.info(f" Trial Leaderboard CSV: {(trials_dir / 'leaderboard.csv').resolve()}")
    logger.info("==================================================")

    # Print summary table
    df_summary = pd.DataFrame(all_records)
    print("\n--- BATCH RUN SUMMARY ---")
    print(df_summary[["trial_id", "feature_mode", "model_type", "seed", "mean_score", "top5_avg_score", "duration_sec", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
