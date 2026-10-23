"""Read a boolean key from config.yaml the way the config schema reads it.

`src/core/config.py` declares `freeze_mode` and
`dataset_governance.require_checksum_match_in_freeze_mode` as `bool` fields, and pydantic
gives such a field a fixed vocabulary. Measured on pydantic 2 by validating each value
through the field itself:

* True:  the strings `true`, `on`, `yes`, `1`, `t`, `y` in any case; `True`; `1`; `1.0`.
* False: the strings `false`, `off`, `no`, `0`, `f`, `n` in any case; `False`; `0`; `0.0`.
* Refused: anything else, including a string with padding whitespace, the strings `"1.0"`
  and `"0.0"`, the empty string, `"none"` and `"null"`, any other number, and a string
  holding a non-ASCII character.

A reader that re-reads the raw YAML instead of going through the schema must not test the
value for truthiness. `yaml.safe_load` returns a real bool for an unquoted `false`, `no` or
`off`, but a QUOTED `"false"` arrives as a string, and `bool("false")` is True, so the
same config.yaml meant "off" to the schema and "on" to that reader. `config_bool` is that
reader's parser.

It reimplements the vocabulary rather than importing pydantic on purpose: pydantic comes
only with the `dev`, `api` and `demo` extras, and `sestrav predict` must keep working on a
core-only install. `tests/test_config_booleans_parse_like_the_schema.py` checks it against
the schema's own fields wherever pydantic is installed, so the two cannot drift.

The one deliberate difference: `None`, which is what a missing key or a bare `key:` gives,
is the reader's `default`. The schema's fields are required, but every reader here treated
an absent value as off, and that is kept.
"""

from __future__ import annotations

_TRUE = frozenset({"true", "on", "yes", "1", "t", "y"})
_FALSE = frozenset({"false", "off", "no", "0", "f", "n"})


def config_bool(key: str, value: object, default: bool = False) -> bool:
    """Return config.yaml's boolean `key`, parsed as `SestravConfig` parses it.

    `key` is used only to name the setting in the error. `None` is `default`. A value
    the schema would refuse raises ValueError naming `key` and the value, rather than
    being read as on or off.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    # bool is handled above, so this is a genuine int or float. 0.5, 2 and -1 are refused.
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        # No guard against non-ASCII is needed: no code point above U+007F lower-cases
        # to any character of these words (checked over the whole of Unicode), so such
        # a string can never match and is refused below, as the schema refuses it.
        spelling = value.lower()
        if spelling in _TRUE:
            return True
        if spelling in _FALSE:
            return False
    raise ValueError(
        f"config.yaml {key}={value!r} is not a boolean: use true/false, yes/no, on/off, "
        "t/f, y/n or 1/0 (in any case), as src/core/config.py's schema accepts"
    )
