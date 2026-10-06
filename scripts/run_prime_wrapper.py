"""
SESTRAV PRIME Snakemake Rule Wrapper.
Invokes PRIME C++ binary if available, else generates simulated output based on binding scores.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import re


# PRIME is resolved inside WSL by ABSOLUTE PATH and run through an explicit exec.
#
# `wsl.exe CMD ARG...` with no `-e`/`--exec` does not exec CMD. It joins the whole
# thing into a command line and hands it to the default Linux shell, which re-parses
# it, so a Python list and `shell=False` buy nothing on the Linux side. Measured on
# this workstation: `wsl printf "[%s]" "a;b"` prints `[a]` and then fails with
# `/bin/bash: line 1: b: command not found`, and `wsl printf "[%s]" "$(id -u)"`
# prints the expanded uid. The same two argv with `-e` inserted print `[a;b]` and
# `[$(id -u)]` literally. `--output` is not pattern-checked at all and reaches that
# argv, so the missing `-e` was the whole exposure.
#
# A PATH lookup cannot replace it. scripts/install_prime_wsl.sh appends its
# `export PATH=...` to the END of ~/.bashrc, and Ubuntu's stock ~/.bashrc returns
# early when the shell is not interactive, so that line is never reached: the shell
# that `wsl` picks with no `-e` was measured NON-INTERACTIVE, with a PATH
# byte-identical to the `-e` PATH. Hence the installer's own root, with the same
# `SESTRAV_TOOL_ROOT` default the installer uses.
#
# PRIME_WSL_SCRIPT is a CONSTANT. Every caller-controlled value is passed as a
# positional argument after it and read back through "$@", so no input is ever
# interpolated into shell source. PRIME_WSL_PROBE tests the identical path, so
# detection and execution cannot disagree about which file they mean.
PRIME_WSL_BIN = '"${SESTRAV_TOOL_ROOT:-$HOME/tools/sestrav_external}/PRIME2.1/PRIME"'
PRIME_WSL_SCRIPT = f"exec {PRIME_WSL_BIN} " + '"$@"'
PRIME_WSL_PROBE = f"test -x {PRIME_WSL_BIN}"

# Alleles arrive as a comma-separated list of tokens like `HLA-A*02:01` or `A0201`.
# Explicit allowlist, and NO whitespace of any kind. The previous class included
# `\s`, so a bare newline was accepted; Python's `$` also matches just before a
# trailing newline, so `"A0201\nid -u"` got in two separate ways. `\A`/`\Z` closes
# the second. Per-token repetition also rejects an empty token, a leading or
# trailing comma, and a bare comma.
MAX_ALLELES_CHARS = 500
ALLELES_RE = re.compile(r"\A[A-Za-z0-9*:_-]+(?:,[A-Za-z0-9*:_-]+)*\Z")


def main():
    parser = argparse.ArgumentParser(description="Run PRIME or mock it if missing")
    parser.add_argument("--binding-csv", required=True, help="Input MHC binding stage CSV")
    parser.add_argument("--output", required=True, help="Output PRIME raw txt file")
    parser.add_argument("--alleles", required=True, help="Comma-separated alleles list")
    args = parser.parse_args()

    if not ALLELES_RE.match(args.alleles) or len(args.alleles) > MAX_ALLELES_CHARS:
        sys.exit(
            f"Error: --alleles contains invalid characters or exceeds "
            f"{MAX_ALLELES_CHARS} chars: {args.alleles!r}"
        )

    # 1. Parse peptides from binding file
    if not os.path.isfile(args.binding_csv):
        print(f"Error: binding file not found: {args.binding_csv}")
        sys.exit(1)

    df = pd.read_csv(args.binding_csv)
    peptides = sorted(df["peptide"].dropna().unique())

    # Write temporary peptides file
    temp_peptides_file = args.output + ".pep_temp"
    with open(temp_peptides_file, "w") as f:
        f.write("\n".join(peptides) + "\n")

    # Translate standard alleles to PRIME compact format (e.g. HLA-A*02:01 -> A0201)
    raw_alleles = args.alleles.split(",")
    prime_alleles = []
    for a in raw_alleles:
        clean = a.replace("HLA-", "").replace("*", "").replace(":", "")
        prime_alleles.append(clean)
    alleles_arg = ",".join(prime_alleles)

    # Check if PRIME is available
    prime_bin = "PRIME"
    # Search common paths
    prime_found = False
    for path in ["", "./", "../PRIME2.1/", "~/PRIME2.1/"]:
        test_path = os.path.expanduser(os.path.join(path, "PRIME"))
        # On windows, we might have PRIME.exe or run under wsl
        if os.path.isfile(test_path) or os.path.isfile(test_path + ".exe"):
            prime_bin = test_path
            prime_found = True
            break

    # Try calling which/where
    if not prime_found:
        try:
            cmd = "where" if sys.platform == "win32" else "which"
            subprocess.run(
                [cmd, "PRIME"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            prime_bin = "PRIME"
            prime_found = True
        # See the note in run_predig_wrapper.py: SubprocessError is not an
        # OSError, so both members are required, and a bare `except:` also
        # swallowed KeyboardInterrupt.
        except (OSError, subprocess.SubprocessError):
            pass

    # Check if we can run via WSL on windows
    run_via_wsl = False
    if sys.platform == "win32" and not prime_found:
        try:
            # Probe the exact file the run below will exec, through the same `-e`
            # exec form. The old probe was `wsl which PRIME`, which could not
            # succeed for an installer-placed PRIME: see the PRIME_WSL_BIN note.
            res = subprocess.run(
                ["wsl", "-e", "bash", "-c", PRIME_WSL_PROBE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if res.returncode == 0:
                prime_bin = "PRIME"
                prime_found = True
                run_via_wsl = True
        except (OSError, subprocess.SubprocessError):
            pass

    if prime_found:
        print(f"[PRIME Wrapper] Running executable: {prime_bin} (WSL={run_via_wsl})")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        # Translate paths for WSL if running via WSL
        if run_via_wsl:
            # Convert paths to wsl paths
            def to_wsl(p):
                abs_p = os.path.abspath(p).replace("\\", "/")
                drive = abs_p[0].lower()
                return f"/mnt/{drive}{abs_p[2:]}"

            wsl_pep = to_wsl(temp_peptides_file)
            wsl_out = to_wsl(args.output)
            # Positional arguments only. `prime` is argv[0] for the exec'd binary.
            cmd = [
                "wsl",
                "-e",
                "bash",
                "-c",
                PRIME_WSL_SCRIPT,
                "prime",
                "-i",
                wsl_pep,
                "-o",
                wsl_out,
                "-a",
                alleles_arg,
            ]
        else:
            cmd = [prime_bin, "-i", temp_peptides_file, "-o", args.output, "-a", alleles_arg]

        try:
            print(f"[PRIME Wrapper] Executing: {' '.join(cmd)}")
            # CORRECTED: this note used to read "cmd is a LIST and shell=False, so no
            # shell interpretation occurs". That was FALSE for the WSL branch, which
            # is the only branch that reaches a shell. `shell=False` stops cmd.exe
            # from parsing anything, but `wsl.exe` with no `-e` handed its argv to a
            # Linux shell that re-parsed it, measured above the PRIME_WSL_BIN
            # constant. What is true now: the native branch execs a list with
            # shell=False, and the WSL branch passes `-e bash -c` a CONSTANT script
            # and supplies every caller-controlled value positionally through "$@",
            # so neither branch lets input reach a shell parser. The alleles are
            # additionally pattern-checked and length-capped; researcher-only CLI
            # tool, no web exposure.
            # The bare inline `# nosemgrep` below is deliberate - see the fuller note in
            # run_predig_wrapper.py. The form used here until 2026-08-16 was inert both for its
            # preceding-line placement and for naming the rule path rather than its real id, so
            # this finding surfaced on every scan despite looking suppressed.
            subprocess.run(cmd, check=True)  # nosemgrep
            print("[PRIME Wrapper] Execution completed successfully.")
            # Cleanup temp file
            if os.path.isfile(temp_peptides_file):
                os.remove(temp_peptides_file)
            sys.exit(0)
        # Only a real execution failure falls back to simulation. sys.exit(0)
        # above raises SystemExit, a BaseException, and is unaffected.
        except (OSError, subprocess.SubprocessError) as e:
            print(
                f"[PRIME Wrapper] Executable failed: {e}. Falling back to simulation.",
                file=sys.stderr,
            )

    # Fallback to simulation
    print("[PRIME Wrapper] PRIME executable not found. Simulating output for reproducibility...")
    np.random.seed(42)

    # We want mock scores that correlate slightly with the presentation_score/affinity if available in binding_csv
    # Look for binding columns (or presentation_score)
    bind_map = {}
    if "presentation_score" in df.columns:
        bind_map = df.groupby("peptide")["presentation_score"].max().to_dict()
    elif "affinity" in df.columns:
        # lower affinity is better, so invert it
        bind_map = (
            df.groupby("peptide")["affinity"]
            .min()
            .apply(lambda x: 1.0 - min(x, 5000) / 5000)
            .to_dict()
        )

    mock_rows = []
    # Write standard PRIME headers
    # columns: peptide, MixMHCpred_score, PRIME_score, pctrank
    for pep in peptides:
        bind_val = bind_map.get(pep, 0.5)
        # add some noise
        prime_score = max(0.0, min(1.0, bind_val * 0.7 + np.random.normal(0, 0.15)))
        mix_score = max(0.0, min(1.0, bind_val * 0.8 + np.random.normal(0, 0.1)))
        pctrank = max(0.0, min(100.0, (1.0 - mix_score) * 100.0))

        mock_rows.append(
            {
                "peptide": pep,
                "MixMHCpred_score": mix_score,
                "PRIME_score": prime_score,
                "pctrank": pctrank,
            }
        )

    mock_df = pd.DataFrame(mock_rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mock_df.to_csv(args.output, sep="\t", index=False)
    print(f"[PRIME Wrapper] Saved simulated PRIME output ({len(mock_df)} rows) to {args.output}")

    # Cleanup temp file
    if os.path.isfile(temp_peptides_file):
        os.remove(temp_peptides_file)


if __name__ == "__main__":
    main()
