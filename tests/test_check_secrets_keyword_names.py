"""Key-material NAMES the secret scan could not reach (2026-10-13 audit, M4).

scripts/check_secrets.py's keyword alternation carries api_key and private_key but
no bare `key`, so `ACCESS_KEY = "<v>"` and `SIGNING_KEY = "<v>"` named nothing it
recognised. Measured at 5e72008d with an 18-character value that clears the floors:
both were missed in a .py file and ACCESS_KEY in a .env file, while the
`API_KEY = "<v>"` control was flagged. The scanner now carries a closed list of five
names; the comment above CREDENTIAL_KEY_MATERIAL records why a `*_KEY` rule is still
refused and why each boundary below exists.

Every name in this module is BUILT from its stem at runtime and every line is
assembled from a template, so the module carries no credential name beside an
operator and the live-tree scans (test_check_secrets.py's
test_the_live_repository_passes_its_own_scan and test_check_secrets_live.py) stay
green over it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"

_STEMS = ("access", "signing", "encryption", "decryption", "hmac")


def _load():
    spec = importlib.util.spec_from_file_location("check_secrets", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _token() -> str:
    # Mixed alphabet, length > 8, entropy > 3.0. Not a real credential.
    return "a8f3k9d2m1q7x4z0b5"


def _spell(stem: str, style: str) -> str:
    if style == "upper":
        return stem.upper() + "_" + "KEY"
    if style == "lower":
        return stem + "_" + "key"
    return stem + "K" + "ey"


def _scan(tmp_path: Path, filename: str, line: str) -> list:
    target = tmp_path / filename
    target.write_text(line + "\n", encoding="utf-8")
    return _load().scan_file(str(target))


# (file, name spelling, template). Each quoted shape reaches CREDENTIAL_KEY_MATERIAL;
# each unquoted one reaches CREDENTIAL_KEY_MATERIAL_BARE, which runs only in a format
# whose unquoted scalar IS the string.
_SHAPES = (
    ("case.py", "upper", '{n} = "{v}"'),
    ("case.toml", "lower", '{n} = "{v}"'),
    ("case.json", "camel", '{{"{n}": "{v}"}}'),
    ("case.json", "lower", '{{"{n}": "{v}"}}'),
    ("case.yml", "lower", '{n}: "{v}"'),
    ("case.yaml", "lower", "{n}: {v}"),
    ("case.yaml", "lower", '"{n}": {v}'),
    ("case.env", "upper", "{n}={v}"),
    ("case.sh", "upper", "export {n}={v}"),
)


@pytest.mark.parametrize("stem", _STEMS)
@pytest.mark.parametrize(("filename", "style", "template"), _SHAPES)
def test_each_key_material_name_is_flagged_in_every_format(
    tmp_path: Path, stem: str, filename: str, style: str, template: str
) -> None:
    line = template.format(n=_spell(stem, style), v=_token())
    assert _scan(tmp_path, filename, line) == [1], f"{filename}: {template}"


def test_prefixed_and_kebab_case_names_are_flagged(tmp_path: Path) -> None:
    """A name may carry any prefix that ends in a separator, as real ones do, and
    YAML and HTTP headers write it in kebab case."""
    cases = (
        ("case.py", "AWS_" + _spell("access", "upper") + ' = "' + _token() + '"'),
        ("case.env", "MY_APP_" + _spell("signing", "upper") + "=" + _token()),
        ("case.py", "self." + _spell("hmac", "lower") + ' = "' + _token() + '"'),
        ("case.yaml", "signing" + "-" + "key: " + _token()),
        ("case.md", "X-Access" + "-" + "Key: " + _token()),
    )
    for filename, line in cases:
        assert _scan(tmp_path, filename, line) == [1], line[:24]


def test_camel_case_humps_are_flagged(tmp_path: Path) -> None:
    """The left boundary admits a camelCase hump, after a lowercase letter or after an
    acronym, so a prefixed camelCase name is reached as the shared alternation reaches
    accessToken."""
    for name in ("aws" + "Access" + "Key", "AWS" + "Signing" + "Key", "my" + "Hmac" + "Key"):
        line = '{"' + name + '": "' + _token() + '"}'
        assert _scan(tmp_path, "case.json", line) == [1], name


def test_a_stem_ending_a_longer_word_is_not_flagged(tmp_path: Path) -> None:
    """ASSIGNING and DESIGNING end in "signing". Without the left boundary both match,
    and a case-insensitive hump would admit the all-caps spelling."""
    names = (
        "AS" + "SIGNING_KEY",
        "DE" + "SIGNING_KEY",
        "as" + "signing_key",
        "as" + "signing" + "Key",
        "RE" + "ENCRYPTION_KEY",
    )
    for name in names:
        assert _scan(tmp_path, "case.py", name + ' = "' + _token() + '"') == [], name
        assert _scan(tmp_path, "case.env", name + "=" + _token()) == [], name


def test_a_run_together_name_is_not_flagged(tmp_path: Path) -> None:
    """`signingkey` is git's own config name (user.signingkey), which per git-config(1)
    names a key (a GPG key specifier, a key file's path or a public key) and never
    holds secret key material, so a separator or a camelCase `Key` is required."""
    cases = (
        ("case.md", "    signing" + "key = " + _token()),
        ("case.ini", "signing" + "key = " + _token()),
        ("case.env", "ACCESS" + "KEY=" + _token()),
        ("case.py", "hmac" + "key" + ' = "' + _token() + '"'),
    )
    for filename, line in cases:
        assert _scan(tmp_path, filename, line) == [], line[:20]


def test_a_name_continuing_past_key_is_not_flagged(tmp_path: Path) -> None:
    """A suffix names an identifier or a file, not the key itself."""
    cases = (
        ("case.py", _spell("access", "upper") + "_ID" + ' = "' + _token() + '"'),
        ("case.py", _spell("signing", "upper") + "_PATH" + ' = "' + _token() + '"'),
        ("case.env", _spell("encryption", "upper") + "_FILE=" + _token()),
    )
    for filename, line in cases:
        assert _scan(tmp_path, filename, line) == [], line[:24]


_PLACEHOLDERS = (
    "<your-access-key>",
    "${CI_PROVIDED_VALUE}",
    "%ENCRYPTION_VALUE_VAR%",
    "{hmac_value_template}",
)


def test_a_quoted_placeholder_is_not_flagged(tmp_path: Path) -> None:
    mod = _load()
    for value in _PLACEHOLDERS:
        # Anti-vacuity: the value clears both floors, so only the placeholder rule
        # can be what keeps the line quiet.
        assert len(value) > 8 and mod.calculate_entropy(value) > 3.0, value
        line = _spell("signing", "upper") + ' = "' + value + '"'
        assert _scan(tmp_path, "case.py", line) == [], value


def test_an_unquoted_placeholder_is_not_flagged(tmp_path: Path) -> None:
    mod = _load()
    for value in _PLACEHOLDERS:
        # The unquoted capture stops before `}`, so check the part it would take.
        captured = value.split("}")[0]
        assert len(captured) > 8 and mod.calculate_entropy(captured) > 3.0, value
        line = _spell("access", "upper") + "=" + value
        assert _scan(tmp_path, "case.env", line) == [], value


def test_the_unquoted_form_runs_only_where_the_shared_one_does(tmp_path: Path) -> None:
    """In Python an unquoted right-hand side is an expression, never a hardcoded
    key; the same text in YAML is the string itself."""
    line = _spell("signing", "lower") + " = settings.signing_material_xyz9"
    assert _scan(tmp_path, "case.py", line) == []
    assert _scan(tmp_path, "case.yaml", line) == [1]


def test_lookup_key_identifiers_and_left_out_names_stay_clean(tmp_path: Path) -> None:
    """Code identifiers that end in KEY, and the candidates left out on meaning.

    If a future change widens the name list to reach one of these, this fails and
    the census recorded in the scanner has to be redone rather than rediscovered."""
    names = (
        "primary" + "_key",
        "sort" + "_key",
        "cache" + "_key",
        "dict" + "_key",
        "partition" + "_key",
        "KEY" + "_COLUMNS",
        "_SELECTED" + "_KEY",
        "master" + "_key",
        "session" + "_key",
        "public" + "_key",
        "consumer" + "_key",
        "kms" + "_key",
        "client" + "_key",
    )
    for name in names:
        assert _scan(tmp_path, "case.py", name + ' = "' + _token() + '"') == [], name
        assert _scan(tmp_path, "case.env", name + "=" + _token()) == [], name


@pytest.mark.parametrize(
    ("filename", "template"),
    (("case.py", '{n} = "{v}"'), ("case.env", "{n}={v}")),
)
def test_key_material_names_keep_the_length_and_entropy_floors(
    tmp_path: Path, filename: str, template: str
) -> None:
    name = _spell("encryption", "upper")
    # Eight characters is not over the length floor; forty repeats fail entropy.
    for value in ("ab12", "ab12cd34", "a" * 40):
        assert _scan(tmp_path, filename, template.format(n=name, v=value)) == [], value
    prose = name + ': "must be rotated every ninety days"'
    assert _scan(tmp_path, "case.yaml", prose) == []
    assert _scan(tmp_path, filename, template.format(n=name, v=_token())) == [1]
