"""`src.verify.structural_gnn` must import cleanly when torch_geometric is absent.

The regression this file exists for, measured 2026-09-21:

`structural_gnn.py` binds `Data`, `Dataset`, `DataLoader`, `GINEConv` and
`global_mean_pool` inside a guarded `try: ... except ImportError: HAS_PYG = False`.
Every RUNTIME use is correctly guarded. One ANNOTATION was not:

    def get(self, idx: int) -> Data:

The module carries no `from __future__ import annotations`, so that annotation is
evaluated at `def` time, and the `def` sits in a class body which executes at module
import. With torch_geometric absent the import therefore raised
`NameError: name 'Data' is not defined` - a try-bound name used in the one position
Python evaluates eagerly.

Two things made it expensive rather than merely wrong:

1. `src/verify/sestrav_evaluator.py` guards the same import with `except ImportError`,
   which CANNOT catch NameError, so the failure propagated and disabled an explicitly
   designed PyG-absent mock-fallback path in that module too.
2. Nothing caught it. ruff passes (F821 cannot fire - `Data` IS bound), mypy passes, and
   `tests/test_dev_extra_runs_the_test_suite.py` skips `ast.Try` nodes by its own stated
   premise that such a guard "degrades on failure rather than raising", which is false
   here. It reached `pip install sestrav`, `pipeline.smk`, and `pip install -e ".[dev]"` -
   the documented developer install - where four test modules failed at COLLECTION, so
   none of their tests ran at all.

Why a subprocess and not monkeypatch: the two existing `HAS_PYG=False` tests in
tests/test_structural_gnn.py patch the flag on an ALREADY-IMPORTED module, so they
cannot see an import-time failure by construction. This defect lives in the import
itself, so it has to be exercised in a fresh interpreter with the package genuinely
unavailable.

`test_the_shadow_really_hides_torch_geometric` is the load-bearing anti-vacuity anchor.
If the shadow silently failed to apply, the import tests below would pass against the
broken code and certify nothing, so the substitution is asserted rather than assumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules whose import must survive torch_geometric being unavailable. The second is
# the amplification: its own `except ImportError` cannot catch a NameError raised by
# the first.
PYG_OPTIONAL_MODULES = [
    "src.verify.structural_gnn",
    "src.verify.sestrav_evaluator",
]


@pytest.fixture(scope="module")
def pyg_shadow(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A PYTHONPATH whose first entry makes `import torch_geometric` raise ImportError."""
    shadow = tmp_path_factory.mktemp("pyg_shadow")
    package = shadow / "torch_geometric"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise ImportError("torch_geometric is shadowed for this test")\n',
        encoding="utf-8",
    )
    return os.pathsep.join([str(shadow), str(REPO_ROOT)])


def _run(code: str, pythonpath: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_shadow_really_hides_torch_geometric(pyg_shadow: str) -> None:
    """Anti-vacuity anchor: prove the substitution applies before relying on it."""
    result = _run(
        "import torch_geometric\n",
        pyg_shadow,
    )
    assert result.returncode != 0, (
        "the shadow did not apply - the real torch_geometric was imported, so every "
        f"other test in this file would certify nothing. stdout={result.stdout!r}"
    )
    assert "ImportError" in result.stderr, result.stderr


@pytest.mark.parametrize("module", PYG_OPTIONAL_MODULES)
def test_module_imports_with_torch_geometric_absent(module: str, pyg_shadow: str) -> None:
    result = _run(f"import {module}\n", pyg_shadow)
    assert result.returncode == 0, (
        f"{module} failed to import with torch_geometric absent. This is the documented "
        f"developer-install path. stderr:\n{result.stderr}"
    )
    assert "NameError" not in result.stderr, result.stderr


def test_has_pyg_is_false_rather_than_raising(pyg_shadow: str) -> None:
    """The guard must DEGRADE, which is what its `except ImportError` promises."""
    result = _run(
        "import src.verify.structural_gnn as m\n"
        "assert m.HAS_PYG is False, m.HAS_PYG\n"
        "print('HAS_PYG', m.HAS_PYG)\n",
        pyg_shadow,
    )
    assert result.returncode == 0, result.stderr
    assert "HAS_PYG False" in result.stdout, result.stdout
