"""Compute the LOO binding-confound decomposition (manuscript Table 3b).

Regenerates results/loo_binding_confound_decomposition.csv from two canonical,
already-committed source CSVs so every cell is reproducible and integrity-bound:

  Regime 1 within-virus, all negatives (real + decoy):
      results/per_virus_eval_v5_mode31.csv  column auc_roc
  Regime 2 within-virus, real negatives only:
      results/per_virus_eval_v5_mode31.csv  column auc_roc_real_neg_only
  Regime 3 cross-virus leave-one-out (real negatives, target held out of train):
      results/loo_cross_virus_v5_clean.csv  column auc_roc

Decoy inflation = R1 - R2 ; transfer gap = R2 - R3, both computed at full float
precision and rounded to 3 decimals (compute-then-round). The three zero-decoy
viruses (HBV/HCV/HPV) carry exactly 0.000 inflation, a built-in mechanism control.

--output has no default: results/loo_binding_confound_decomposition.csv is a
git-tracked artifact, so a bare invocation prints the table without writing
anything rather than silently rewriting it.

Reproduce:  python scripts/compute_loo_binding_confound.py --output results/loo_binding_confound_decomposition.csv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os

import pandas as pd

CANON = ["CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"]
PV_SRC = "results/per_virus_eval_v5_mode31.csv"
LOO_SRC = "results/loo_cross_virus_v5_clean.csv"
TRACKED_OUTPUT = "results/loo_binding_confound_decomposition.csv"


def compute_decomposition() -> pd.DataFrame:
    pv = pd.read_csv(PV_SRC).set_index("virus")
    loo = pd.read_csv(LOO_SRC).set_index("test_virus")

    rows = []
    # Full-precision parallel record, kept ONLY to compute the Mean row.
    # See the mean_row comment below for why this exists.
    exact: dict[str, list[float]] = {
        "within_all_neg": [],
        "within_real_neg": [],
        "loo_cross_virus": [],
        "decoy_inflation": [],
        "transfer_gap": [],
    }
    for v in CANON:
        r1 = float(pv.loc[v, "auc_roc"])
        r2 = float(pv.loc[v, "auc_roc_real_neg_only"])
        r3 = float(loo.loc[v, "auc_roc"])
        decoy_frac = float(pv.loc[v, "n_neg_decoy"]) / (
            float(pv.loc[v, "n_neg_real"]) + float(pv.loc[v, "n_neg_decoy"])
        )
        exact["within_all_neg"].append(r1)
        exact["within_real_neg"].append(r2)
        exact["loo_cross_virus"].append(r3)
        exact["decoy_inflation"].append(r1 - r2)
        exact["transfer_gap"].append(r2 - r3)
        rows.append(
            {
                "virus": v,
                "within_all_neg": round(r1, 3),       # Regime 1
                "within_real_neg": round(r2, 3),      # Regime 2
                "loo_cross_virus": round(r3, 3),      # Regime 3
                "decoy_inflation": round(r1 - r2, 3),  # R1 - R2
                "transfer_gap": round(r2 - r3, 3),     # R2 - R3
                "decoy_frac_neg": round(decoy_frac, 3),
            }
        )

    tab = pd.DataFrame(rows)
    # Mean is computed from the FULL-PRECISION values, then rounded once -
    # matching this module's stated compute-then-round contract.
    #
    # Corrected 2026-08-15: this previously averaged tab[...], i.e. the
    # already-rounded per-virus column, making it round-then-mean and quietly
    # contradicting the docstring. It produced identical output on the current
    # corpus (verified: all five means agree to 3dp either way), so nothing
    # published was ever wrong - which is exactly what made it easy to miss.
    # It was a latent divergence waiting for a corpus change to surface it, and
    # the failure mode would have been a Mean cell drifting one unit in the last
    # place for no reason visible in the table.
    def _mean3(column: str) -> float:
        values = exact[column]
        return round(sum(values) / len(values), 3)

    mean_row = {
        "virus": "Mean",
        "within_all_neg": _mean3("within_all_neg"),
        "within_real_neg": _mean3("within_real_neg"),
        "loo_cross_virus": _mean3("loo_cross_virus"),
        "decoy_inflation": _mean3("decoy_inflation"),
        "transfer_gap": _mean3("transfer_gap"),
        "decoy_frac_neg": "",
    }
    return pd.concat([tab, pd.DataFrame([mean_row])], ignore_index=True)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sidecar(output_path: str) -> str:
    """Record input digests alongside the artifact.

    Added 2026-08-15. This artifact previously shipped with NO sidecar, so the
    integrity harness reported it as "no checksum recorded" and nothing verified
    it. Writing one only became safe once the CSV writer above pinned LF - see
    that comment for why a sidecar written against CRLF bytes would have
    recorded a digest no Linux checkout could reproduce.
    """
    sidecar_path = f"{output_path}.provenance.json"
    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "script": "scripts/compute_loo_binding_confound.py",
        "artifact": output_path,
        "sha256": _sha256_file(output_path),
        "inputs": {
            PV_SRC: _sha256_file(PV_SRC),
            LOO_SRC: _sha256_file(LOO_SRC),
        },
        "viruses": CANON,
        "rounding": "compute-then-round at 3dp, including the Mean row",
    }
    with open(sidecar_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return sidecar_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the LOO binding-confound decomposition (manuscript Table 3b)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Output CSV path (optional). No default: {TRACKED_OUTPUT} is a "
            "git-tracked artifact, so this script refuses to guess a destination "
            "- omit this flag to print the table without writing anything."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = compute_decomposition()
    print(out.to_string(index=False))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # lineterminator pinned to LF (added 2026-08-15, D24-resid). Without it
        # this wrote CRLF on Windows, and the tracked artifact IS currently in
        # that state: it carries an `eol=lf` .gitattributes pin, so `git status`
        # reads clean, while its working-tree bytes differ from the LF blob by
        # 11 CRLF pairs. That is the trap - the pin normalises on check-in and
        # therefore HIDES the divergence rather than preventing it. Any sha256
        # taken of the working tree would be a Windows-only value that no Linux
        # clone or CI checkout could reproduce, which is precisely why the
        # sidecar below could not safely be written before this line existed.
        out.to_csv(args.output, index=False, lineterminator="\n")
        sidecar = _write_sidecar(args.output)
        print(f"\nwrote {args.output}")
        print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
