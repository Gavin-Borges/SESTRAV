"""How a provenance or run-metadata writer records a filesystem path: never as a workstation path.

The repo's privacy mandate forbids writing an absolute workstation path into any
repo file, tracked or gitignored. :func:`relative_source` is the reduction a
writer applies to a caller-supplied path before recording it. A path is covered
only where a writer calls this function on it (``git grep -n relative_source``
finds the calls). ``src.artifact_integrity._relative_to_project_root`` is a
separate, older reduction that does not call it; it produces the ``artifact``
field of ``write_provenance_sidecar``, ``model_provenance_fields``'
``model_path`` and ``binding_matrix_provenance_fields``' ``binding_matrix_path``.

The rule, in the order it is applied, to the value with any leading and trailing
whitespace removed (so a space before ``https://`` or ``D:\\`` cannot hide either):

* a value holding a character that is not printable (a line break, a tab, a
  NUL) is reduced as below, because such a character can put a second path
  after the first;
* a URL whose scheme is on an explicit allowlist (``http``, ``https``, ``ftp``,
  ``ftps``, ``s3``, ``gs``) and whose host is a host name or a bracketed IP
  literal, with an optional numeric port, keeps its scheme, its host and its
  path. Any ``user:password@`` before the host, and any query string or fragment,
  is removed. A URL whose host is empty or is not of that form
  (``https:///C:/x.csv``, ``https://C:/x.csv``) is reduced as below. No
  allowlisted scheme is one letter, so a drive spelling such as ``C://`` does
  not match;
* a value already marked ``external:`` keeps the mark, and anything after it is
  reduced as below;
* a ``file:`` URI is reduced as below;
* an absolute path inside the checkout becomes its repo-relative POSIX form,
  unless a second pass would not keep that form as it is (it names a workstation
  location, say), in which case it is reduced as below;
* anything else that names a workstation location - an absolute path outside the
  checkout, a drive letter (including a drive-relative ``C:x``), a component that
  ends in a colon (the scheme of a URL off the allowlist such as ``smb://``, or
  what is left of any URL once it has been resolved as a path), a UNC or
  root-relative path, a leading ``~``, a ``Users`` or ``home`` component, or a
  relative path that escapes upwards (judged textually, so the working directory
  cannot change the answer) - is reduced as below. A component is judged with
  its own surrounding whitespace removed, so ``data/ D:/x`` names a drive;
* a relative path holding an ``@`` anywhere (a ``user@host``, or a
  ``user:password@`` with no scheme before it), or a ``:`` before its first
  separator (where scp and rsync read ``[user@]host:path``), is reduced as
  below. A plain ``@`` in a file's name is reduced too, because the text cannot
  tell ``a@b.csv`` from a user ``a`` at a host ``b.csv``;
* any other relative path is kept exactly as given, separators included. The
  test reads the text alone: such a path is not a workstation location as
  defined above, holds no ``@`` and has no ``:`` before its first separator,
  but it is not otherwise known to name no user or host. A component that
  merely equals a username, as in ``data/<name>/x.csv``, or a percent-encoded
  ``%40`` is kept, for example. Committed provenance already records some kept
  paths with Windows separators (``data\\raw\\ctl_all.csv``), which rewriting
  would change on the next regeneration.

The reduction is ``external:`` plus at most the value's last component, read from
the TEXT after separators are normalized and ``..`` segments are collapsed, never
from the filesystem, so ``..`` on its own cannot pick up the name of whatever
directory the process happens to run in. Before that component is read, a leading
URL scheme (``smb://``, ``file:``) is removed, and a ``?`` or ``#`` ends the text:
it starts a query string or fragment, and if a separator follows it the last
component may lie inside that query or fragment, so none is kept. The component
directly after a leading ``//`` is an authority (a URL's ``user:password@host``,
or a UNC server), never a file's name. A last component that cannot be a file's
own name is dropped, leaving ``external:`` alone: empty, ``.``, ``..``, that
authority, a name holding a ``:`` or ``@`` (a bare drive or scheme such as ``C:``
or ``smb:``, a ``host:port``, a ``user:password@host``), a ``~`` form, a name
holding a character that is not printable, or a name sitting directly under a
``Users`` or ``home`` directory, which is a username.

Every value this returns is a fixed point of it, so a caller that already
sanitized its own sources is not changed by a second pass.
"""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path
from typing import overload

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_PREFIX = "external:"

# Only these schemes name a location that is not on this workstation. A scheme
# outside the list is not passed through.
URL_SCHEMES = ("http", "https", "ftp", "ftps", "s3", "gs")
_URL = re.compile(r"(?:" + "|".join(URL_SCHEMES) + r")://", re.IGNORECASE)
_FILE_URI = re.compile(r"file:", re.IGNORECASE)
_DRIVE = re.compile(r"[A-Za-z]:")
# A whole component ending in a colon: the scheme of a URL off the allowlist
# ("smb:" in "smb://host/x"), or what a URL leaves once a caller has resolved it
# as a path ("https:" in "https:/user:pw@host/x" inside the checkout).
_SCHEME_REMNANT = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")
# A URL scheme the reduction removes before reading the last component: one of
# two or more characters followed by "//" (a one-letter one is a drive), or "file:".
_URL_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]+:(?=//)|file:", re.IGNORECASE)
# The host an allowlisted URL may keep: a name holding no whitespace, separator,
# ":", "?", "#", "@" or bracket, or a bracketed IP literal; then an optional
# numeric port. An empty host does not match.
_HOST = re.compile(r"(?:[^\s/\\:?#@\[\]]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]+)?")
# Directory names whose immediate child is a user's home directory.
_HOME_PARENTS = frozenset({"users", "home"})


def _stripped_components(posix_text: str) -> str:
    """``posix_text`` with the surrounding whitespace of each component removed."""
    return "/".join(part.strip() for part in posix_text.split("/"))


def _components(posix_text: str) -> list[str]:
    return [part for part in posix_text.split("/") if part]


def _names_a_workstation_location(posix_text: str) -> bool:
    """True if a separator-normalized path names a location on this workstation."""
    posix_text = _stripped_components(posix_text)
    if posix_text.startswith(("/", "~")):  # POSIX/root-relative, UNC (//host), home
        return True
    normalized = posixpath.normpath(posix_text) if posix_text else ""
    if normalized == ".." or normalized.startswith("../"):
        return True
    return any(
        _DRIVE.match(part) or _SCHEME_REMNANT.fullmatch(part) or part.lower() in _HOME_PARENTS
        for part in _components(posix_text)
    )


def _names_a_user_or_remote_host(posix_text: str) -> bool:
    """True if a separator-normalized relative path holds an ``@`` anywhere, or a
    ``:`` before its first separator, where scp and rsync read ``[user@]host:path``."""
    return "@" in posix_text or ":" in posix_text.split("/", 1)[0]


def _external(text: str) -> str:
    """``external:`` plus the last component of ``text``, when that is a file's name."""
    posix_text = text.strip().replace("\\", "/")
    while prefix := _URL_PREFIX.match(posix_text):
        posix_text = posix_text[prefix.end() :]
    marks = [index for index in (posix_text.find("?"), posix_text.find("#")) if index != -1]
    if marks:
        mark = min(marks)
        posix_text = "" if "/" in posix_text[mark:] else posix_text[:mark]
    posix_text = _stripped_components(posix_text)
    after_authority = posix_text.startswith("//") and not posix_text.startswith("///")
    parts = _components(posixpath.normpath(posix_text) if posix_text else "")
    name = parts[-1] if parts else ""
    # A drive-relative "C:x.csv" carries its drive glued to the name.
    if _DRIVE.match(name):
        name = name[2:].strip()
    parent = parts[-2].lower() if len(parts) > 1 else ""
    if (
        name in ("", ".", "..")
        or (after_authority and len(parts) == 1)
        or ":" in name
        or "@" in name
        or name.startswith("~")
        or not name.isprintable()
        or parent in _HOME_PARENTS
    ):
        name = ""
    return EXTERNAL_PREFIX + name


def _url_without_credentials_or_query(url: str, scheme_end: int) -> str | None:
    """``url`` with any ``user:password@``, query string and fragment removed, or
    None when what is left of its authority is not a host ``_HOST`` accepts."""
    rest = url[scheme_end:]
    authority_end = len(rest)
    for mark in "/?#":
        index = rest.find(mark)
        if index != -1:
            authority_end = min(authority_end, index)
    host = rest[:authority_end].rpartition("@")[2]
    if not _HOST.fullmatch(host):
        return None
    path = rest[authority_end:]
    for mark in "?#":
        path = path.split(mark, 1)[0]
    return url[:scheme_end] + host + path.rstrip()


@overload
def relative_source(
    source: str | os.PathLike[str], root: str | os.PathLike[str] | None = ...
) -> str: ...


@overload
def relative_source(source: None, root: str | os.PathLike[str] | None = ...) -> None: ...


@overload
def relative_source(source: object, root: str | os.PathLike[str] | None = ...) -> object: ...


def relative_source(source: object, root: str | os.PathLike[str] | None = None) -> object:
    """Return how a writer records ``source``: never as a workstation path.

    ``source`` is a string or path-like; anything else (``None``, a dict) is returned
    unchanged, because only a caller knows which fields of a structured value are
    paths. ``root`` is the checkout a path is made relative to, and defaults to the
    one this module lives in.
    """
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if not isinstance(source, str):
        return source
    text = source.strip()
    if not text.isprintable():
        return _external(text)
    url = _URL.match(text)
    if url:
        kept = _url_without_credentials_or_query(text, url.end())
        return kept if kept is not None else _external(text)
    if text[: len(EXTERNAL_PREFIX)].lower() == EXTERNAL_PREFIX:
        return _external(text[len(EXTERNAL_PREFIX) :])
    if _FILE_URI.match(text):
        return _external(text)

    posix_text = text.replace("\\", "/")
    if os.path.isabs(text):
        # Both sides in one resolved form, so a symlinked or dotted spelling of
        # either cannot move a file in or out of the checkout.
        path = os.path.realpath(text)
        base = os.path.realpath(os.fspath(root) if root is not None else PROJECT_ROOT)
        try:
            inside = os.path.commonpath([base, path]) == base
        except ValueError:  # on Windows, a path on a different drive from the repo
            inside = False
        if inside:
            relative = os.path.relpath(path, base).replace(os.sep, "/")
            # Kept only if a second pass keeps it as it is, so the result is a
            # fixed point: it names no workstation location and is not read as
            # anything else (a padded name, an "external:" or "file:" directory).
            if relative_source(relative, root) == relative:
                return relative
        return _external(text)
    if _names_a_workstation_location(posix_text) or _names_a_user_or_remote_host(posix_text):
        return _external(text)
    return text
