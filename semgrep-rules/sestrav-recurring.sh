#!/usr/bin/env bash

# ruleid: sestrav-preserve-pipeline-status
pytest -q | tail -n 10

# ok: sestrav-preserve-pipeline-status
pytest -q

# ruleid: sestrav-require-git-glob-pathspec
git ls-files 'tests/**/test_*.py'

# ok: sestrav-require-git-glob-pathspec
git ls-files ':(glob)tests/**/test_*.py'

# ruleid: sestrav-require-git-glob-pathspec
git -C "$ROOT" ls-files 'scripts/**/*.py'

# ok: sestrav-require-git-glob-pathspec
git -C "$ROOT" ls-files ':(glob)scripts/**/*.py'

# ruleid: sestrav-count-cr-bytes-not-lines
grep -c $'\r' artifact.txt

# ruleid: sestrav-count-cr-bytes-not-lines
grep -cP '\r' artifact.txt

# ruleid: sestrav-count-cr-bytes-not-lines
grep -c -P '\r' artifact.txt

# ok: sestrav-count-cr-bytes-not-lines
tr -cd '\r' < artifact.txt | wc -c

# Commented-out commands must not fire. Three rules above exclude comment lines,
# and the fixture checker fails on any finding these lines draw.
#   pytest -q | tail -n 10
#   git ls-files 'tests/**/test_*.py'
#   grep -c $'\r' artifact.txt
