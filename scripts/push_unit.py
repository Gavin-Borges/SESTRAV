#!/usr/bin/env python3
"""Prepare one landing unit from its worktree; optionally push it and open its PR.

    python scripts/push_unit.py --unit UNIT_ID --worktree PATH --title "PR TITLE"
        --body-file PATH [--reviewer SESSION_ID] [--prep-only] [--allow-primary]
        [--kill-merge-loops | --ignore-merge-loops] [--allow-behind] [--no-fetch]
        [--base main] [--remote origin] [--signoff-name "Gavin Borges"]

This is the assistant's half of landing, and it never merges. The owner runs
`land-unit UNIT_ID` (scripts/owner/land-unit.ps1), which re-verifies the record
written here and arms auto-merge pinned to the exact commit it names.

Refused, in this order:
 1. Merge loops. Any running process whose command line runs `gh pr merge`,
    names a REST merge endpoint, or names a GraphQL merge mutation could merge
    the unit before the owner looks at it. Reported and refused (exit 4) unless
    --kill-merge-loops (terminate them, with their child processes) or
    --ignore-merge-loops.
 2. The worktree. It must be a git work tree; a LINKED worktree, not the
    primary checkout, unless --allow-primary; on a branch, neither detached nor
    the base branch; with nothing modified or untracked; with no merge, rebase,
    cherry-pick or revert in progress; and with core.hooksPath naming a
    scripts/hooks directory that holds pre-push. core.hooksPath may be absolute
    and point into the primary checkout, whose hook then runs for this push;
    that is reported, not refused.
 3. The commits in <remote>/<base>..HEAD, after fetching <base> (--no-fetch
    skips that). There must be at least one, and the branch must not be behind
    the base: the ruleset requires an up-to-date branch, so a behind PR cannot
    merge until its head changes (--allow-behind overrides). A non-merge commit
    must be authored and committed by --signoff-name, carry a
    "Signed-off-by: <that name> <email>" trailer, and have a good SSH signature
    (%G? is G and the commit holds an SSH signature header). A merge commit
    must be signed: SSH and good, or a signature GitHub made for its own
    update-branch merge. Merges carry no sign-off check, as the DCO workflow
    reads `git rev-list --no-merges`. Every message must pass the
    commit-msg hook's rules, mirrored line by line below, plus no en-dash.
 4. The PR title and body. The same message rules apply to both. The title also
    may not hold a double quote, a backtick or a line break: Windows PowerShell
    5.1 mangles a native argument that holds a double quote, and land-unit
    passes the title to gh.
 5. --reviewer naming this session. The author never grades its own work.

Then it writes _local/landing/<UNIT_ID>.json in the PRIMARY checkout. Paths in
it are relative to that checkout, so no workstation path is written. Unless
--prep-only, it pushes HEAD to <remote>/<branch> (never forced; pre-push runs),
confirms the remote ref with ls-remote, finds or opens the PR, records its
number, and prints "Ready to land: land-unit <UNIT_ID>".

Exit codes: 0 ready; 2 usage; 3 a check failed; 4 merge loops found;
5 the push or PR step failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXIT_OK, EXIT_USAGE, EXIT_CHECK, EXIT_LOOPS, EXIT_PUSH = 0, 2, 3, 4, 5
UNIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_SIGNOFF = "Gavin Borges"

# Mirrors scripts/hooks/commit-msg, applied line by line as grep applies it.
BARE_AI = ("ChatGPT", "Copilot", "Generated with")
ATTRIBUTION = re.compile(
    r"(generated|authored|created|written|made)\s+(with|by)\s+"
    r"(claude|anthropic|codex|cursor|copilot|chatgpt|gpt)"
    r"|claude\s+code"
    r"|signed-off-by\s*:\s*(claude|anthropic|codex|cursor|copilot|chatgpt|gpt)"
    r"|\bby\s+anthropic\b|anthropic\s+ai\b",
    re.IGNORECASE,
)
COAUTHOR = re.compile(r"co-authored-by", re.IGNORECASE)
# Built from code points so this file never holds the characters it rejects.
EM_DASH, EN_DASH = chr(0x2014), chr(0x2013)
SIGNOFF = re.compile(r"^Signed-off-by:[ \t]*(?P<name>.+?)[ \t]*<(?P<email>[^<>\s]+)>[ \t]*$")

MERGE_LOOP = re.compile(
    r"\bgh(?:\.exe)?[\"']?\s+(?:\S+\s+){0,6}?pr\s+(?:\S+\s+){0,6}?merge\b"
    r"|pulls/[^/\s]+/merge\b"
    r"|\b(?:mergePullRequest|enablePullRequestAutoMerge|enqueuePullRequest)\b",
    re.IGNORECASE,
)
IN_PROGRESS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD",
               "rebase-merge", "rebase-apply")
# Discovery variables a hook or a parent git exports; they would point every
# git call below at the wrong repository.
GIT_DISCOVERY = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX",
                 "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY")


class UnitError(Exception):
    def __init__(self, message: str, code: int = EXIT_CHECK):
        super().__init__(message)
        self.code = code


# ---- subprocess helpers -----------------------------------------------------

def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in GIT_DISCOVERY}
    env["GH_PROMPT_DISABLED"] = "1"
    return env


def run(argv: list[str], cwd: Path, *, check: bool = True,
        capture: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        argv, cwd=str(cwd), env=_env(), capture_output=capture, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise UnitError(f"{' '.join(argv[:3])} ... exited {result.returncode}: {detail[:400]}")
    return result


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return run(["git", *args], cwd, check=check).stdout.strip()


# ---- 1. merge loops -----------------------------------------------------------

def _process_table() -> list[tuple[int, int, str]]:
    """(pid, parent pid, command line) for every process this user can see."""
    try:
        import psutil
    except ImportError:
        psutil = None
    rows: list[tuple[int, int, str]] = []
    if psutil is not None:
        for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
            try:
                cmd = " ".join(proc.info["cmdline"] or [])
            except (psutil.Error, TypeError):
                continue
            rows.append((proc.info["pid"], proc.info["ppid"] or 0, cmd))
        return rows
    if os.name == "nt":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,"
             "CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        ).stdout
        data = json.loads(out or "[]")
        for item in data if isinstance(data, list) else [data]:
            rows.append((int(item["ProcessId"]), int(item["ParentProcessId"] or 0),
                         item.get("CommandLine") or ""))
        return rows
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True,
                         check=False).stdout
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2:
            rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return rows


def find_merge_loops(rows: list[tuple[int, int, str]] | None = None,
                     me: int | None = None) -> list[tuple[int, str]]:
    """Processes whose command line merges a PR, excluding this one and its ancestors."""
    rows = _process_table() if rows is None else rows
    me = os.getpid() if me is None else me
    parent = {pid: ppid for pid, ppid, _ in rows}
    mine, cursor = set(), me
    while cursor and cursor not in mine:
        mine.add(cursor)
        cursor = parent.get(cursor, 0)
    return [(pid, cmd) for pid, _, cmd in rows
            if pid not in mine and "push_unit.py" not in cmd and MERGE_LOOP.search(cmd)]


def kill_tree(pid: int) -> None:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            family = proc.children(recursive=True) + [proc]
        except psutil.Error:
            return
        for member in family:
            try:
                member.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(family, timeout=3)
        for member in alive:
            try:
                member.kill()
            except psutil.Error:
                pass
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


# ---- 2. the worktree ---------------------------------------------------------------

def worktree_state(path: Path) -> dict:
    probe = run(["git", "rev-parse", "--show-toplevel"], path, check=False)
    if probe.returncode != 0:
        raise UnitError(f"{path} is not inside a git work tree")
    top = Path(probe.stdout.strip()).resolve()
    git_dir = Path(git(top, "rev-parse", "--path-format=absolute", "--git-dir")).resolve()
    common = Path(git(top, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    branch = run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], top, check=False)
    hooks = run(["git", "config", "--get", "core.hooksPath"], top, check=False)
    in_progress = []
    for name in IN_PROGRESS:
        marker = git(top, "rev-parse", "--path-format=absolute", "--git-path", name)
        if Path(marker).exists():
            in_progress.append(name)
    return {
        "top": top,
        "linked": git_dir != common,
        "primary": common.parent,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "head": git(top, "rev-parse", "HEAD"),
        "porcelain": git(top, "status", "--porcelain=v1", "--untracked-files=normal").splitlines(),
        "in_progress": in_progress,
        "hooks_path": hooks.stdout.strip() if hooks.returncode == 0 else None,
    }


def hooks_dir(state: dict) -> Path | None:
    value = state["hooks_path"]
    if not value:
        return None
    path = Path(value)
    return (path if path.is_absolute() else state["top"] / path).resolve()


def worktree_problems(state: dict, *, allow_primary: bool, base: str) -> list[str]:
    problems = []
    if not state["linked"] and not allow_primary:
        problems.append(
            f"{state['top']} is the primary checkout. Land from a linked worktree "
            "(python .claude/tools/coord.py wt new <slug>), or pass --allow-primary."
        )
    if state["branch"] is None:
        problems.append("HEAD is detached; check out the unit's branch first")
    elif state["branch"] == base:
        problems.append(f"the worktree is on {base}; a unit lands from its own branch")
    if state["porcelain"]:
        shown = "; ".join(state["porcelain"][:5])
        more = f" (+{len(state['porcelain']) - 5} more)" if len(state["porcelain"]) > 5 else ""
        problems.append(f"the worktree is not clean: {shown}{more}")
    if state["in_progress"]:
        problems.append("an operation is in progress: " + ", ".join(state["in_progress"]))
    hooks = hooks_dir(state)
    if hooks is None:
        problems.append("core.hooksPath is not set, so no pre-push hook would run")
    elif hooks.name != "hooks" or hooks.parent.name != "scripts" or not (hooks / "pre-push").is_file():
        problems.append(f"core.hooksPath is {state['hooks_path']}, not a scripts/hooks "
                        "directory holding pre-push")
    return problems


# ---- 3 and 4. commits, title and body ---------------------------------------------

def message_problems(text: str, label: str) -> list[str]:
    """The commit-msg hook's rules, line by line, plus the en-dash."""
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        for marker in BARE_AI:
            if marker.lower() in line.lower():
                problems.append(f"{label} line {number}: AI marker '{marker}'")
        match = ATTRIBUTION.search(line)
        if match:
            problems.append(f"{label} line {number}: AI attribution '{match.group(0)}'")
        if COAUTHOR.search(line):
            problems.append(f"{label} line {number}: co-author trailer")
        if EM_DASH in line:
            problems.append(f"{label} line {number}: em-dash (U+2014)")
        if EN_DASH in line:
            problems.append(f"{label} line {number}: en-dash (U+2013)")
    return problems


def title_problems(title: str) -> list[str]:
    problems = message_problems(title, "PR title")
    if not title.strip() or title != title.strip():
        problems.append("PR title is empty or has leading or trailing space")
    if len(title) > 256:
        problems.append("PR title is longer than 256 characters")
    for char, name in (('"', "a double quote"), ("`", "a backtick"), ("\n", "a line break"),
                       ("\r", "a carriage return")):
        if char in title:
            problems.append(f"PR title holds {name}")
    return problems


def signoff_names(message: str) -> list[str]:
    names = []
    for line in message.splitlines():
        match = SIGNOFF.match(line.strip())
        if match:
            names.append(match.group("name"))
    return names


def commit_problems(rec: dict, signoff_name: str) -> list[str]:
    """rec: sha, parents (list), sig (%G?), header (raw commit header), author,
    committer, message."""
    short = rec["sha"][:10]
    problems = message_problems(rec["message"], f"commit {short}")
    ssh = "gpgsig -----BEGIN SSH SIGNATURE-----" in rec["header"]
    gpg = "gpgsig -----BEGIN PGP SIGNATURE-----" in rec["header"]
    if len(rec["parents"]) > 1:
        github_merge = rec["committer"] == "GitHub" and gpg
        if not ((rec["sig"] == "G" and ssh) or github_merge):
            problems.append(f"commit {short}: merge commit is not signed (%G? = {rec['sig']})")
        return problems
    if rec["author"] != signoff_name:
        problems.append(f"commit {short}: author is '{rec['author']}', not '{signoff_name}'")
    if rec["committer"] != signoff_name:
        problems.append(f"commit {short}: committer is '{rec['committer']}', not '{signoff_name}'")
    if signoff_name not in signoff_names(rec["message"]):
        problems.append(f"commit {short}: no 'Signed-off-by: {signoff_name} <email>' trailer")
    if not ssh:
        problems.append(f"commit {short}: no SSH signature")
    elif rec["sig"] != "G":
        problems.append(f"commit {short}: SSH signature does not verify (%G? = {rec['sig']})")
    return problems


def list_commits(top: Path, base_ref: str) -> list[dict]:
    out = run(["git", "log", "--format=%H%x1f%P%x1f%G?%x1f%an%x1f%cn%x1f%B%x1e",
               f"{base_ref}..HEAD"], top).stdout
    commits = []
    for chunk in out.split("\x1e"):
        chunk = chunk.lstrip("\r\n")
        if not chunk.strip():
            continue
        sha, parents, sig, author, committer, message = chunk.split("\x1f", 5)
        raw = run(["git", "cat-file", "commit", sha], top).stdout
        commits.append({
            "sha": sha, "parents": parents.split(), "sig": sig, "author": author,
            "committer": committer, "message": message.rstrip("\n"),
            "header": raw.split("\n\n", 1)[0],
        })
    return commits


# ---- 5. the landing record ------------------------------------------------------------

def current_session() -> str | None:
    return os.environ.get("CLAUDE_CODE_SESSION_ID") or os.environ.get("SESTRAV_SESSION") or None


def reviewer_problems(reviewer: str | None, session: str | None) -> list[str]:
    if not reviewer or not session:
        return []
    a, b = reviewer.lower(), session.lower()
    if a.startswith(b[:8]) or b.startswith(a):
        return [f"--reviewer {reviewer} is this session; the author never grades its own work"]
    return []


def relative_to_primary(path: Path, primary: Path) -> str:
    try:
        return Path(os.path.relpath(path, primary)).as_posix()
    except ValueError as exc:  # another drive on Windows
        raise UnitError(f"{path} is not on the same drive as {primary}; "
                        "the landing record stores relative paths only") from exc


def write_record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(tmp, path)


# ---- the PR step -------------------------------------------------------------

def gh_json(top: Path, *args: str):
    return json.loads(run(["gh", *args], top).stdout or "null")


def find_prs(top: Path, branch: str) -> list[dict]:
    fields = "number,state,url,headRefName,headRefOid,baseRefName,isDraft"
    prs = gh_json(top, "pr", "list", "--head", branch, "--state", "all", "--json", fields,
                  "--limit", "20") or []
    if prs:
        return prs
    # An empty answer is not proof of absence: cross-check every open PR.
    every_open = gh_json(top, "pr", "list", "--state", "open", "--json", fields,
                         "--limit", "200") or []
    return [p for p in every_open if p.get("headRefName") == branch]


def push_and_open(top: Path, rec: dict, body_file: Path, title: str) -> dict:
    branch, remote, head = rec["branch"], rec["remote"], rec["expected_head_sha"]
    print(f"[push_unit] Pushing {head[:10]} to {remote}/{branch} (pre-push runs)...", flush=True)
    pushed = run(["git", "push", "--set-upstream", remote, f"HEAD:refs/heads/{branch}"], top,
                 check=False, capture=False)
    if pushed.returncode != 0:
        raise UnitError(f"git push exited {pushed.returncode}; nothing was forced", EXIT_PUSH)
    remote_line = git(top, "ls-remote", remote, f"refs/heads/{branch}")
    if remote_line.split("\t", 1)[0] != head:
        raise UnitError(f"{remote}/{branch} is {remote_line[:40] or 'missing'}, not {head}", EXIT_PUSH)

    prs = find_prs(top, branch)
    open_prs = [p for p in prs if p.get("state") == "OPEN"]
    if open_prs:
        pr = open_prs[0]
        print(f"[push_unit] WARNING: PR #{pr['number']} already exists for {branch}; "
              "updating its title and body.")
        run(["gh", "pr", "edit", str(pr["number"]), "--title", title, "--body-file",
             str(body_file)], top)
    elif any(p.get("state") == "MERGED" for p in prs):
        number = next(p["number"] for p in prs if p.get("state") == "MERGED")
        raise UnitError(f"{branch} was already merged as PR #{number}", EXIT_PUSH)
    elif prs:
        raise UnitError(f"PR #{prs[0]['number']} for {branch} is closed; reopening it is "
                        "the owner's call", EXIT_PUSH)
    else:
        created = run(["gh", "pr", "create", "--base", rec["base"], "--head", branch,
                       "--title", title, "--body-file", str(body_file)], top, check=False)
        if created.returncode != 0:
            prs = [p for p in find_prs(top, branch) if p.get("state") == "OPEN"]
            if not prs:
                raise UnitError("gh pr create failed: " + (created.stderr or "").strip()[:400],
                                EXIT_PUSH)
            print(f"[push_unit] WARNING: PR #{prs[0]['number']} appeared while creating; "
                  "another actor opened it. Using it.")
        pr = (prs or [None])[0] if created.returncode != 0 else None
        if pr is None:
            pr = gh_json(top, "pr", "view", created.stdout.strip().splitlines()[-1], "--json",
                         "number,state,url,headRefName,headRefOid,baseRefName,isDraft")
    for _attempt in range(6):  # GitHub can lag the push by a few seconds
        if pr.get("headRefOid") == head:
            break
        time.sleep(2)
        pr = gh_json(top, "pr", "view", str(pr["number"]), "--json",
                     "number,state,url,headRefName,headRefOid,baseRefName,isDraft")
    if pr.get("headRefOid") != head:
        print(f"[push_unit] WARNING: PR #{pr['number']} reports head {pr.get('headRefOid')}, "
              f"not {head}; land-unit will refuse until they agree.")
    if pr.get("baseRefName") != rec["base"]:
        print(f"[push_unit] WARNING: PR #{pr['number']} targets {pr.get('baseRefName')}, "
              f"not {rec['base']}.")
    return pr


# ---- main -----------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--reviewer")
    parser.add_argument("--prep-only", action="store_true")
    parser.add_argument("--allow-primary", action="store_true")
    loops = parser.add_mutually_exclusive_group()
    loops.add_argument("--kill-merge-loops", action="store_true")
    loops.add_argument("--ignore-merge-loops", action="store_true")
    parser.add_argument("--allow-behind", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--base", default="main")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--signoff-name", default=DEFAULT_SIGNOFF)
    return parser.parse_args(argv)


def _fail(problems: list[str]) -> None:
    for problem in problems:
        print(f"[push_unit] REFUSED: {problem}", file=sys.stderr)
    raise UnitError(f"{len(problems)} check(s) failed")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not UNIT_RE.match(args.unit):
        print("[push_unit] --unit: letters, digits, '.', '_' or '-', at most 64", file=sys.stderr)
        return EXIT_USAGE
    body_file = Path(args.body_file).resolve()
    worktree = Path(args.worktree).resolve()
    try:
        loops = find_merge_loops()
        if loops:
            for pid, cmd in loops:
                print(f"[push_unit] merge loop: pid {pid}: {cmd[:160]}", file=sys.stderr)
            if args.kill_merge_loops:
                for pid, _ in loops:
                    kill_tree(pid)
                print(f"[push_unit] terminated {len(loops)} merge loop process(es)")
            elif not args.ignore_merge_loops:
                raise UnitError("merge loops are running; stop them or pass --kill-merge-loops",
                                EXIT_LOOPS)

        state = worktree_state(worktree)
        top = state["top"]
        os.chdir(top)
        problems = worktree_problems(state, allow_primary=args.allow_primary, base=args.base)
        hooks = hooks_dir(state)
        if hooks is not None and not str(hooks).startswith(str(top)):
            print(f"[push_unit] note: core.hooksPath is {hooks}; that checkout's pre-push "
                  "runs for this push")

        if not args.no_fetch:
            fetched = run(["git", "fetch", "--quiet", args.remote, args.base], top, check=False)
            if fetched.returncode != 0:
                problems.append(f"git fetch {args.remote} {args.base} failed: "
                                + (fetched.stderr or "").strip()[:200])
        base_ref = f"{args.remote}/{args.base}"
        counts = git(top, "rev-list", "--left-right", "--count", f"{base_ref}...HEAD").split()
        behind, ahead = int(counts[0]), int(counts[1])
        if ahead == 0:
            problems.append(f"HEAD has no commits that {base_ref} lacks; nothing to land")
        if behind and not args.allow_behind:
            problems.append(f"the branch is {behind} commit(s) behind {base_ref}; merge or "
                            f"rebase onto it first (the ruleset requires an up-to-date branch)")
        for rec in list_commits(top, base_ref):
            problems.extend(commit_problems(rec, args.signoff_name))

        problems.extend(title_problems(args.title))
        if not body_file.is_file():
            problems.append(f"--body-file {body_file} does not exist")
        else:
            try:
                problems.extend(message_problems(body_file.read_text(encoding="utf-8"), "PR body"))
            except UnicodeDecodeError:
                problems.append("PR body is not UTF-8")
        problems.extend(reviewer_problems(args.reviewer, current_session()))
        if problems:
            _fail(problems)

        primary = state["primary"]
        record_path = primary / "_local" / "landing" / f"{args.unit}.json"
        previous = {}
        if record_path.is_file():
            previous = json.loads(record_path.read_text(encoding="utf-8"))
            if previous.get("branch") not in (None, state["branch"]):
                _fail([f"unit {args.unit} is already recorded for branch {previous['branch']}"])
        rec = {
            "schema": 1,
            "unit_id": args.unit,
            "branch": state["branch"],
            "base": args.base,
            "remote": args.remote,
            "worktree_path": relative_to_primary(top, primary),
            "pr_title": args.title,
            "pr_body_path": relative_to_primary(body_file, primary),
            "expected_head_sha": state["head"],
            "commits": ahead,
            "reviewer": args.reviewer,
            "prepared_by": current_session(),
            "prepared_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pushed": False,
            "pr_number": previous.get("pr_number"),
            "pr_url": previous.get("pr_url"),
        }
        write_record(record_path, rec)
        print(f"[push_unit] recorded {record_path.relative_to(primary).as_posix()}: "
              f"{state['branch']} at {state['head'][:10]} ({ahead} commit(s))")
        if args.prep_only:
            print(f"[push_unit] Prepared, not pushed. Owner: land-unit {args.unit}")
            return EXIT_OK

        pr = push_and_open(top, rec, body_file, args.title)
        rec.update({"pushed": True, "pr_number": pr["number"], "pr_url": pr["url"],
                    "pushed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        write_record(record_path, rec)
        print(f"[push_unit] PR #{pr['number']}: {pr['url']}")
        print(f"Ready to land: land-unit {args.unit}")
        return EXIT_OK
    except UnitError as exc:
        print(f"[push_unit] {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
