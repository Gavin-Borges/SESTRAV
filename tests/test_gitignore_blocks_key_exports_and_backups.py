""".gitignore stops key exports, credential JSON, editor state and backups.

The security block already ignored `*.key`, `*.pem`, `*.p12`, `*.pfx`, `*.jks` and
the four OpenSSH default key names, but not the other common shapes a private key
or token takes on disk: a PuTTY key (`.ppk`), an ASCII-armoured PGP export
(`gpg --export-secret-keys --armor` writes `.asc` by convention), a binary keyring
(`.gpg`), a cloud service-account or OAuth client JSON, or a `token.txt`. Editor
directories (`.vscode/`, `.idea/`) hold launch configurations that carry
environment variables, and backup and merge leftovers (`*.bak`, `*.orig`, `*.rej`,
Vim swap files) are copies of files whose current version may have been cleaned.
Each of these was freely committable.

`--no-index` is used because none of these files exists in the tree: the question
is what the rules say, not what is on disk.

The negative controls matter as much as the positives. A bare `core` rule, the
obvious way to ignore a core dump, also matches `src/core/`, which holds three
tracked modules, and would silently ignore any new file added there; the rule is
anchored to the repository root instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

MUST_BE_IGNORED = [
    "deploy.ppk",
    "keys/signing-key.asc",
    "pubring.gpg",
    "credentials.json",
    "config/credentials.json",
    "client_secret_1234.apps.googleusercontent.com.json",
    "service-account.json",
    "gcp-service-account.json",
    "token.txt",
    "scripts/token.txt",
    ".vscode/launch.json",
    ".idea/workspace.xml",
    "config.yaml.bak",
    "pipeline.smk.orig",
    "pipeline.smk.rej",
    ".config.yaml.swp",
    ".config.yaml.swo",
    "core",
    "core.12345",
    "src/core.4711",
]

MUST_NOT_BE_IGNORED = [
    "README.md",
    "src/core/config.py",
    "src/core/a_module_added_later.py",
    "docs/core.md",
    "tests/test_core_config.py",
    "environments/requirements-pip-bootstrap.txt",
]


def _ignored(path: str) -> bool:
    done = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", path],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    # 0 = ignored, 1 = not ignored, anything else = git failed.
    assert done.returncode in (0, 1), done.stderr
    return done.returncode == 0


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_key_export_credential_and_backup_names_are_ignored(path: str) -> None:
    assert _ignored(path), f"{path} is committable; .gitignore should block it"


@pytest.mark.parametrize("path", MUST_NOT_BE_IGNORED)
def test_tracked_sources_are_not_swept_up(path: str) -> None:
    assert not _ignored(path), f"{path} reads as ignored; a rule above is too broad"
