"""
Run PredIG Docker in batches (5000-row limit) and merge outputs.

Usage:
    python scripts/run_predig_batched.py [--batch-size 4800]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch PredIG Docker runs")
    parser.add_argument(
        "--input",
        default="results/external_predig_input_recombinant.csv",
    )
    parser.add_argument(
        "--output",
        default="results/external_tool_outputs/predig_path_output.csv",
    )
    parser.add_argument("--batch-size", type=int, default=4800)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--image",
        default=os.environ.get("SESTRAV_PREDIG_IMAGE", "bsceapm/predig:latest"),
        help="PredIG Docker image reference (prefer a digest-pinned ref in production)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.repo_root)
    os.chdir(root)

    df = pd.read_csv(args.input)
    out_dir = os.path.dirname(args.output) or "results/external_tool_outputs"
    os.makedirs(out_dir, exist_ok=True)

    batch_dir = os.path.join(out_dir, "predig_batches")
    os.makedirs(batch_dir, exist_ok=True)

    results_mount = os.path.abspath("results").replace("\\", "/")
    if results_mount[1] == ":":
        results_mount = f"/{results_mount[0].lower()}{results_mount[2:]}"
    proteome_mount = os.path.abspath("data/proteomes").replace("\\", "/")
    if proteome_mount[1] == ":":
        proteome_mount = f"/{proteome_mount[0].lower()}{proteome_mount[2:]}"

    parts = []
    n_batches = (len(df) + args.batch_size - 1) // args.batch_size
    for i in range(n_batches):
        chunk = df.iloc[i * args.batch_size : (i + 1) * args.batch_size]
        in_name = f"predig_batch_{i+1}_input.csv"
        out_name = f"predig_batch_{i+1}_output.csv"
        in_path = os.path.join("results", "external_tool_outputs", "predig_batches", in_name)
        out_path = os.path.join("results", "external_tool_outputs", "predig_batches", out_name)
        chunk.to_csv(in_path, index=False)

        docker_in = f"/predig/external_tool_outputs/predig_batches/{in_name}"
        docker_out = f"/predig/external_tool_outputs/predig_batches/{out_name}"
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.abspath('results')}:/predig",
            "-v",
            f"{os.path.abspath('data/proteomes')}:/uniprot",
            args.image,
            docker_in,
            "--output",
            docker_out,
            "--model",
            "path",
            "--type",
            "recombinant",
        ]
        print(f"[predig-batch] Running batch {i+1}/{n_batches} ({len(chunk)} rows)")
        subprocess.run(cmd, check=True)
        parts.append(pd.read_csv(out_path))

    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(args.output, index=False)
    print(f"[predig-batch] Merged {len(merged)} rows -> {args.output}")


if __name__ == "__main__":
    main()
