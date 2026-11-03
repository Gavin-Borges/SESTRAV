"""tools/update_dependencies.py resolves every lockfile against PyPI alone.

Measured on uv 0.12.0 by recompiling the ci-ruff lock through the tool with a
dead index (http://127.0.0.1:9/simple) injected one route at a time. Each of
these reached it: UV_DEFAULT_INDEX, UV_INDEX_URL, UV_EXTRA_INDEX_URL, UV_INDEX,
UV_FIND_LINKS, UV_CONFIG_FILE, a user-level uv.toml, a UV_CONSTRAINT file
carrying an --extra-index-url or --find-links line, and a UV_OVERRIDE file
carrying an --extra-index-url line. UV_CUSTOM_COMPILE_COMMAND rewrote the lock
header, and UV_EXCLUDE_NEWER and UV_WORKING_DIR made the run fail. Two measured
asymmetries shape the fix: --index-url outranks UV_INDEX_URL only, while
--default-index outranks UV_INDEX_URL and UV_DEFAULT_INDEX; and UV_NO_CONFIG=1
stops a discovered uv.toml but not an explicit UV_CONFIG_FILE. uv adds variables
like these from release to release, so the tool keeps a short list of
operational UV_* variables and drops every other one. uv is not a repo
dependency, so these tests pin the argv and the environment the tool gives uv
rather than running it.
"""

import subprocess
import tomllib

from tools import update_dependencies
from tools.update_dependencies import (
    DEFAULT_INDEX,
    LOCK_SPECS,
    UV_ENV_KEPT,
    build_command,
    compile_env,
)

DEAD_INDEX = "http://127.0.0.1:9/simple"

# Every flag that names where uv fetches from, in each spelling uv accepts.
INDEX_FLAGS = ("--index-url", "-i", "--extra-index-url", "--index", "--find-links", "-f")

# Variables measured or documented to change what uv resolves or writes. Stated
# here rather than derived from the module, so the module cannot keep one by
# accident and pass anyway.
RESOLUTION_ROUTES = (
    "UV_BUILD_CONSTRAINT",
    "UV_CONFIG_FILE",
    "UV_CONSTRAINT",
    "UV_CUSTOM_COMPILE_COMMAND",
    "UV_DEFAULT_INDEX",
    "UV_EXCLUDE_NEWER",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX",
    "UV_INDEX_STRATEGY",
    "UV_INDEX_URL",
    "UV_INSECURE_HOST",
    "UV_OFFLINE",
    "UV_OVERRIDE",
    "UV_PRERELEASE",
    "UV_RESOLUTION",
    "UV_TORCH_BACKEND",
    "UV_WORKING_DIR",
)


def _uv_settings(env):
    """The UV_* names in env, other than the kept ones and UV_NO_CONFIG."""
    allowed = {*UV_ENV_KEPT, "UV_NO_CONFIG"}
    return sorted(n for n in env if n.upper().startswith("UV_") and n.upper() not in allowed)


def test_pypi_is_the_default_index():
    assert DEFAULT_INDEX == "https://pypi.org/simple"


def test_only_cache_network_and_progress_settings_are_kept():
    assert set(UV_ENV_KEPT) == {
        "UV_CACHE_DIR",
        "UV_HTTP_TIMEOUT",
        "UV_NATIVE_TLS",
        "UV_NO_CACHE",
        "UV_NO_PROGRESS",
        "UV_SYSTEM_CERTS",
    }
    assert not set(UV_ENV_KEPT) & set(RESOLUTION_ROUTES)


def test_every_spec_names_pypi_as_its_default_index_exactly_once():
    for spec in LOCK_SPECS:
        command = build_command(spec)
        assert command.count("--default-index") == 1, spec.name
        assert command[command.index("--default-index") + 1] == DEFAULT_INDEX, spec.name


def test_no_spec_names_any_other_index():
    for spec in LOCK_SPECS:
        command = build_command(spec)
        assert not set(INDEX_FLAGS) & set(command), spec.name


def test_upgrades_keep_the_default_index():
    for spec in LOCK_SPECS:
        for command in (
            build_command(spec, upgrade_package="pillow"),
            build_command(spec, upgrade_all=True),
        ):
            assert command[command.index("--default-index") + 1] == DEFAULT_INDEX, spec.name


def test_compile_env_drops_every_resolution_route_and_disables_config_discovery():
    env = compile_env({name: DEAD_INDEX for name in RESOLUTION_ROUTES})
    assert env == {"UV_NO_CONFIG": "1"}


def test_compile_env_drops_a_uv_variable_it_has_never_heard_of():
    env = compile_env({"UV_ADDED_IN_A_LATER_RELEASE": DEAD_INDEX})
    assert env == {"UV_NO_CONFIG": "1"}


def test_compile_env_matches_names_without_regard_to_case():
    environ = {
        "uv_extra_index_url": DEAD_INDEX,
        "Uv_Constraint": "elsewhere/constraints.txt",
        "uv_no_config": "0",
        "uv_cache_dir": "cache",
    }
    env = compile_env(environ)
    assert _uv_settings(env) == []
    assert [name for name in env if name.upper() == "UV_NO_CONFIG"] == ["UV_NO_CONFIG"]
    assert env["UV_NO_CONFIG"] == "1"
    assert env["uv_cache_dir"] == "cache"


def test_compile_env_keeps_everything_else():
    kept = {
        "PATH": "/usr/bin",
        "HTTPS_PROXY": "http://proxy.invalid:3128",
        "SSL_CERT_FILE": "certs.pem",
        **{name: "1" for name in UV_ENV_KEPT},
    }
    env = compile_env({**kept, "UV_INDEX": DEAD_INDEX, "UV_OVERRIDE": "o.txt"})
    assert env == {**kept, "UV_NO_CONFIG": "1"}


def test_compile_env_leaves_the_callers_mapping_alone():
    environ = {"UV_INDEX_URL": DEAD_INDEX, "PATH": "/usr/bin"}
    compile_env(environ)
    assert environ == {"UV_INDEX_URL": DEAD_INDEX, "PATH": "/usr/bin"}


def test_compile_env_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", DEAD_INDEX)
    monkeypatch.setenv("SESTRAV_INDEX_PROBE", "kept")
    env = compile_env()
    assert _uv_settings(env) == []
    assert env["SESTRAV_INDEX_PROBE"] == "kept"


def test_every_compile_runs_in_the_scrubbed_environment(monkeypatch):
    # --target reaches every spec, the two seeded image specs included, which
    # compile in a scratch directory through their own subprocess call.
    monkeypatch.setattr(update_dependencies, "uv_version", lambda: "uv 0.12.0")
    for name in RESOLUTION_ROUTES:
        monkeypatch.setenv(name, DEAD_INDEX)
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", runner)
    assert update_dependencies.main(["--target", "pillow"]) == 0
    assert len(calls) == len(LOCK_SPECS)
    seeded = [kwargs for kwargs in calls if kwargs["cwd"] != update_dependencies.REPO_ROOT]
    assert len(seeded) == len(update_dependencies.SEEDED_SPECS)
    for kwargs in calls:
        assert _uv_settings(kwargs["env"]) == []
        assert kwargs["env"]["UV_NO_CONFIG"] == "1"


def test_the_repo_carries_no_uv_configuration_that_uv_no_config_would_hide():
    # UV_NO_CONFIG=1 ignores every discovered uv.toml and [tool.uv] table, the
    # repo's own included. None exists today; adding one means deciding first
    # whether the tool should read it.
    root = update_dependencies.REPO_ROOT
    assert not (root / "uv.toml").exists()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "uv" not in pyproject.get("tool", {})
