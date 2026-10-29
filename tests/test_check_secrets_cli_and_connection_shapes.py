"""Credential shapes scripts/check_secrets.py could not see, and their near misses.

Each positive case below was planted with an 18-character mixed-alphabet value
that clears the scanner's length and entropy floors, and each was missed by the
scanner at e653f140 while the plain `api_key = "<v>"` control was flagged:

- a `pass` name (`DB_PASS`, a JSON `"pass"` key), quoted or, in a bare-value
  format, unquoted: the keyword group carries `password`, `passwd` and `pwd`
  but not `pass`;
- a `Password=` or `PWD=` element inside an ADO.NET or ODBC connection string,
  where the value follows `=` with no quote of its own;
- a password handed to a command on its command line: `--password <v>` (the
  `=` form was already caught), `mysql -p<v>`, `curl -u user:<v>` and
  `twine upload -p <v>`;
- an R assignment, `api_key <- "<v>"`, which `.qmd` and `.md` files carry.

The negatives pin what must stay quiet: placeholders and shell variables after a
flag, `bypass` and `passport` (names that only contain `pass`), and a bare
`mysql -p`, which prompts for the password instead of carrying one.

Payloads are assembled at runtime so this module carries no credential-shaped
line of its own; the tree scan reads it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"

# Mixed alphabet, length 18, entropy > 3.0. Not a real credential.
V = "a8f3k9d2m1q7x4z0b5"
PW = "pass" + "word"


def _load():
    spec = importlib.util.spec_from_file_location("check_secrets", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scan(tmp_path: Path, name: str, line: str) -> list[int]:
    target = tmp_path / name
    target.write_text("# context\n" + line + "\n", encoding="utf-8")
    return _load().scan_file(str(target))


FLAGGED = [
    pytest.param("settings.py", "DB_" + "PASS = " + '"' + V + '"', id="py-DB_PASS-quoted"),
    pytest.param("config.json", '{"pa' + 'ss": "' + V + '"}', id="json-pass-key"),
    pytest.param("deploy.yml", "db_pa" + "ss: " + V, id="yml-db_pass-bare"),
    pytest.param(".env", "SMTP_PA" + "SS=" + V, id="dotenv-SMTP_PASS-bare"),
    pytest.param(
        "db.py",
        'CONN = "Server=db;User Id=sa;' + PW.capitalize() + "=" + V + ';"',
        id="py-ado-connection-string",
    ),
    pytest.param(
        "db.py",
        'conn = "DRIVER={x};UID=sa;PW' + "D=" + V + '"',
        id="py-odbc-PWD-last-element",
    ),
    pytest.param("backup.sh", "mysqldump --" + PW + " " + V + " db", id="sh-flag-space"),
    pytest.param(
        "ci.yml", "run: tool --" + PW + " '" + V + "' --verbose", id="yml-flag-space-quoted"
    ),
    pytest.param("backup.sh", "mysql -u root -p" + V + " db", id="sh-mysql-attached-p"),
    pytest.param("fetch.sh", "curl -u admin:" + V + " https://x.example/api", id="sh-curl-user"),
    pytest.param(
        "ci.yml", "run: curl --user 'admin:" + V + "' https://x.example", id="yml-curl-user"
    ),
    pytest.param("release.sh", "twine upload -u __token__ -p " + V + " dist/*", id="sh-twine-p"),
    pytest.param("report.qmd", "api_" + "key <- " + '"' + V + '"', id="qmd-r-arrow"),
    pytest.param("notes.md", "tok" + "en <<- " + "'" + V + "'", id="md-r-superassign"),
]


@pytest.mark.parametrize(("name", "line"), FLAGGED)
def test_missed_shape_is_now_flagged(tmp_path: Path, name: str, line: str) -> None:
    assert _scan(tmp_path, name, line) == [2]


QUIET = [
    # Placeholders and variables after a flag are not credentials.
    pytest.param("USAGE.md", "tool --" + PW + " <" + PW.upper() + "_HERE>", id="md-placeholder"),
    pytest.param("run.sh", "tool --" + PW + ' "$DB_' + PW.upper() + '"', id="sh-variable"),
    pytest.param("ci.yml", "run: tool --" + PW + " ${{ secrets.DB }}", id="yml-actions-secret"),
    pytest.param("run.sh", "curl -u admin:$API_SECRET_VALUE https://x.example", id="sh-curl-var"),
    # A bare -p prompts; mysql reads the password from the terminal.
    pytest.param("run.sh", "mysql -u root -p " + "database_name_long", id="sh-mysql-prompt"),
    # Names that merely contain "pass".
    pytest.param("rules.py", 'bypass_mode = "' + V + '"', id="py-bypass"),
    pytest.param("people.py", 'passport_id = "' + V + '"', id="py-passport"),
    # A keyword argument is code, not a connection-string element.
    pytest.param("db.py", "connect(" + PW + "=" + "credentials_from_vault())", id="py-kwarg"),
    # R comparison against a negative number is not an assignment.
    pytest.param("calc.qmd", "if (token < -1) stop('bad')", id="qmd-r-comparison"),
    # Prose. "documentation" clears both floors (11+ characters, entropy 3.24), so an
    # unquoted lone `pass:` rule would turn the gate red on this repo's own Markdown,
    # which writes "first pass:" and "second pass:" routinely.
    pytest.param("CHANGELOG.md", "Second pass: documentation", id="md-prose-pass-colon"),
    pytest.param("notes.txt", "pass: reproducibility", id="txt-prose-pass-colon"),
]


@pytest.mark.parametrize(("name", "line"), QUIET)
def test_near_miss_stays_quiet(tmp_path: Path, name: str, line: str) -> None:
    assert _scan(tmp_path, name, line) == []
