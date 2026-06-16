"""Subprocess coverage bootstrap.

Some tests exercise modules by launching them as child processes
(e.g. ``python src/data_curation_qc.py``). coverage.py cannot see across a
process boundary unless each child starts its own measurement. Python imports
``sitecustomize`` automatically at interpreter startup when it is importable,
so placing this directory on ``PYTHONPATH`` makes every child process call
``coverage.process_startup()``.

That function is a no-op unless ``COVERAGE_PROCESS_START`` points at a coverage
config file, so this import is safe to leave on the path for normal runs. When
the env var is set (see the CI "Run tests with coverage" step), each child
writes its own ``.coverage.<host>.<pid>`` data file, which pytest-cov then
combines with the parent's data via ``parallel = true``.
"""

try:
    import coverage

    coverage.process_startup()
except Exception:  # pragma: no cover - coverage is a test-only dependency
    # Never let coverage bootstrap break an actual program run.
    pass
