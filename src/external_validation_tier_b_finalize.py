"""
Tier B governance: manifests, SHA256 bundle, provenance (does not touch Tier A raw).

Usage:
    python -m src.external_validation_tier_b_recovery --run-dir <tierB_run>
    python -m src.external_validation_tier_b_finalize --run-dir <tierB_run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from src.external_validation_finalize import build_artifact_manifest, sha256_file


def build_tier_b_provenance(run_dir: str, repo_root: str) -> dict:
    prefilter = os.path.join(repo_root, "results", "external_tier_b_prefilter_manifest.json")
    meta = {
        "run_id": os.path.basename(run_dir.rstrip("/\\")),
        "scope": "tierB",
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "protocol": "H2_ISSR_evaluation_protocol.md §3 (gold-standard recovery in top 10%/25%)",
        "prefilter": "top 2000 peptides/virus by binding_max union 15 canonical GS epitopes",
        "tools": ["SESTRAV RF", "Binding-only", "PredIG-Path", "PRIME 2.1"],
    }
    raw = os.path.join(run_dir, "raw")
    for name, key in [
        ("predig_path_output.csv", "predig"),
        ("prime21_output.txt", "prime"),
    ]:
        path = os.path.join(raw, name)
        if os.path.isfile(path):
            df = (
                pd.read_csv(path, sep="\t", comment="#")
                if name.endswith(".txt")
                else pd.read_csv(path)
            )
            pep_col = next(
                (c for c in df.columns if c.lower() in ("peptide", "seq")),
                df.columns[0],
            )
            meta[key] = {
                "path": f"raw/{name}",
                "sha256": sha256_file(path),
                "n_rows": len(df),
                "n_unique_peptides": int(df[pep_col].nunique()),
            }
    if os.path.isfile(prefilter):
        meta["prefilter_manifest_sha256"] = sha256_file(prefilter)
    rec = os.path.join(run_dir, "processed", "tier_b_gold_standard_recovery.csv")
    if os.path.isfile(rec):
        rdf = pd.read_csv(rec)
        meta["recovery_summary"] = {
            "n_rows": len(rdf),
            "scopes": sorted(rdf["scope"].unique().tolist()) if "scope" in rdf.columns else [],
        }
    return meta


def append_tier_b_report_section(
    report_path: str,
    run_dir: str,
    recovery_csv: str,
) -> None:
    if not os.path.isfile(recovery_csv):
        return
    df = pd.read_csv(recovery_csv)
    combined = df[df["scope"] == "tierB_combined"] if "scope" in df.columns else df
    lines = [
        "",
        "## Tier B — gold-standard recovery (finalized)",
        "",
        f"Run directory: `{os.path.basename(run_dir)}`",
        f"Finalized: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Tool | Top 10% recovery | Top 25% recovery | GS in pool |",
        "|------|------------------|------------------|------------|",
    ]
    for _, row in combined.iterrows():
        tool = row.get("method", row.get("tool", ""))
        r10 = row.get("gs_recovery_top10", "")
        r25 = row.get("gs_recovery_top25", "")
        if r10 != "":
            r10 = f"{float(r10) * 100:.1f}%"
        if r25 != "":
            r25 = f"{float(r25) * 100:.1f}%"
        gs = row.get("gs_found", "")
        lines.append(f"| {tool} | {r10} | {r25} | {gs} |")
    lines.append("")
    lines.append(
        "Source: `processed/tier_b_gold_standard_recovery.csv` in Tier B run dir. "
        "PRIME included when `raw/prime21_output.txt` present."
    )
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            body = f.read()
        marker = "## Tier B — gold-standard recovery"
        if marker in body:
            start = body.index(marker)
            end = body.find("\n## ", start + 1)
            if end == -1:
                end = len(body)
            body = body[:start] + "\n".join(lines) + body[end:]
        else:
            body = body.rstrip() + "\n" + "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(body)


def run_tier_b_finalize(run_dir: str, results_dir: str = "results") -> dict:
    run_dir = os.path.abspath(run_dir)
    repo_root = os.path.abspath(results_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)

    manifests = os.path.join(run_dir, "manifests")
    os.makedirs(manifests, exist_ok=True)

    provenance = build_tier_b_provenance(run_dir, repo_root)
    prov_path = os.path.join(manifests, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    prefilter_src = os.path.join(repo_root, "external_tier_b_prefilter_manifest.json")
    if os.path.isfile(prefilter_src):
        dest = os.path.join(manifests, "tier_b_prefilter_manifest.json")
        if not os.path.isfile(dest):
            import shutil

            shutil.copy2(prefilter_src, dest)

    recovery_csv = os.path.join(run_dir, "processed", "tier_b_gold_standard_recovery.csv")
    legacy: List[str] = []
    report = os.path.join(results_dir, "external_benchmark_comparison.md")
    if os.path.isfile(report):
        legacy.append(report)

    artifact_manifest = build_artifact_manifest(run_dir, legacy)
    art_path = os.path.join(manifests, "artifact_sha256.json")
    with open(art_path, "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    status = {
        "run_id": provenance["run_id"],
        "status": "done",
        "finalized_utc": provenance["finalized_utc"],
        "has_prime": "prime" in provenance,
        "has_predig": "predig" in provenance,
        "artifact_count": len(artifact_manifest),
    }
    status_path = os.path.join(manifests, "tier_b_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    log_path = os.path.join(run_dir, "logs", "commands.log")
    if os.path.isfile(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n# Tier B finalize {provenance['finalized_utc']}\n"
                f"python -m src.external_validation_tier_b_finalize --run-dir {run_dir}\n"
            )

    append_tier_b_report_section(report, run_dir, recovery_csv)

    print(f"[tier-b-finalize] provenance -> {prov_path}")
    print(f"[tier-b-finalize] artifacts -> {art_path} ({len(artifact_manifest)} files)")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier B governance finalize")
    parser.add_argument(
        "--run-dir",
        default="results/external_tool_outputs/extval_20260520_1750_gb_tierB",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    run_tier_b_finalize(args.run_dir, args.results_dir)


if __name__ == "__main__":
    main()
