"""Attachments: review material passed by reference (DESIGN-submit-poll.md §17).

Both `second_opinion` and `submit_second_opinion` accept `attachment_paths`,
local files the *server* reads and splices into the upstream prompt after
`summary`. This module is the guardrail layer (§17.4): it is the first place
the server reads the filesystem at a model's direction, so every path goes
through the root allowlist, the basename denylist, the regular-file and
UTF-8 checks and the size cap before a single byte is read into a prompt.

Every failure is an `AttachmentError`, which the tool layer maps to
`invalid_input` — never `internal_error`. Nothing here logs; the caller logs
sizes and digests, never content.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePath

# Basenames that are never attachable, whatever the roots say (§17.4,
# "defense in depth — the roots do the real work"). Matched case-insensitively
# against the basename of the path as given *and* as resolved. Any dotfile is
# denied too (which also covers `.env*`).
DENYLIST_PATTERNS: tuple[str, ...] = (
    "config*.json",
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "*.p12",
    "*.pfx",
)

# How many hex characters of the digest go into log lines.
DIGEST_LOG_CHARS = 12


class AttachmentError(Exception):
    """An attachment was refused. Always surfaces as `invalid_input`."""


@dataclass(frozen=True)
class Attachment:
    """One loaded attachment: content plus the metadata that is logged and
    echoed. `name` is the basename the caller used; it is what appears in the
    prompt delimiters, the log lines and the result envelope."""

    name: str
    bytes: int
    sha256: str
    content: str

    @property
    def digest(self) -> str:
        return self.sha256[:DIGEST_LOG_CHARS]

    def echo(self) -> dict[str, object]:
        """The `{name, bytes}` shape carried by result envelopes (§17.4)."""
        return {"name": self.name, "bytes": self.bytes}


def render_attachment(attachment: Attachment) -> str:
    """The §17.3 block for one attachment, delimiters exact:

        --- FILE: <basename> (<n> bytes) ---
        <content>
        --- END FILE: <basename> ---

    Content is spliced byte-exact; a single newline is added only when the
    file does not already end with one, so the closing delimiter starts on
    its own line.
    """
    body = attachment.content
    if not body.endswith("\n"):
        body += "\n"
    return (
        f"--- FILE: {attachment.name} ({attachment.bytes} bytes) ---\n"
        f"{body}"
        f"--- END FILE: {attachment.name} ---"
    )


def paths_are_case_insensitive() -> bool:
    """Windows filesystems compare paths case-insensitively; containment must
    too, or `C:\\Root\\x` would escape a root configured as `c:\\root`."""
    return os.name == "nt"


def is_strictly_inside(
    candidate: PurePath, root: PurePath, *, case_insensitive: bool | None = None
) -> bool:
    """True when `candidate` is a proper descendant of `root`.

    Compares path *components*, never string prefixes, so a sibling that
    shares a prefix (`/a/bc/x` against root `/a/b`) is outside. Both sides are
    expected to be already resolved (symlinks followed). On Windows the
    comparison is case-insensitive and the drive is a component like any
    other, so a same-named directory on another drive is outside.
    """
    if case_insensitive is None:
        case_insensitive = paths_are_case_insensitive()

    def parts(p: PurePath) -> tuple[str, ...]:
        raw = p.parts
        return tuple(x.lower() for x in raw) if case_insensitive else tuple(raw)

    c, r = parts(candidate), parts(root)
    return len(c) > len(r) and c[: len(r)] == r


def is_denied_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("."):
        return True
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in DENYLIST_PATTERNS)


def resolve_roots(roots: list[str]) -> list[Path]:
    """Resolve configured roots with symlinks followed. A root that does not
    exist is kept (it simply contains nothing) — config warns about it."""
    return [Path(r).expanduser().resolve() for r in roots if r and r.strip()]


def load_attachments(
    paths: list[str] | None,
    roots: list[str],
    max_total_bytes: int,
) -> list[Attachment]:
    """Validate and read `paths` under the §17.4 guardrails, in order.

    Two passes: every path is resolved, contained, name-checked, type-checked
    and sized *before* any content is read, so the size cap fails fast on
    `stat()` alone; then the files are read and decoded. Raises
    `AttachmentError` on the first violation.
    """
    if paths is None:
        return []
    if not isinstance(paths, list) or any(
        not isinstance(p, str) or not p.strip() for p in paths
    ):
        raise AttachmentError(
            "`attachment_paths` must be a list of non-empty file path strings."
        )
    if not paths:
        return []

    if not roots:
        raise AttachmentError(
            "attachment_paths is disabled: no attachment roots are configured. "
            "Set `attachment_roots` in config.json (a list of directories whose "
            "files may be attached) or LLM_SECOND_OPINION_ATTACHMENT_ROOTS "
            f"(directories separated by {os.pathsep!r}), then restart the server."
        )
    resolved_roots = resolve_roots(roots)

    # Pass 1: guard and size, reading no content.
    staged: list[tuple[str, Path, str, int]] = []
    for raw in paths:
        given = Path(raw).expanduser()
        try:
            resolved = given.resolve(strict=True)
        except FileNotFoundError:
            raise AttachmentError(f"attachment {raw!r} does not exist.") from None
        except (OSError, RuntimeError) as e:
            raise AttachmentError(f"attachment {raw!r} could not be resolved: {e}") from None

        if not any(is_strictly_inside(resolved, root) for root in resolved_roots):
            raise AttachmentError(
                f"attachment {raw!r} is outside the configured attachment_roots "
                f"(checked after resolving symlinks). Allowed roots: "
                f"{[str(r) for r in resolved_roots]}."
            )

        name = given.name or resolved.name
        if is_denied_name(name) or is_denied_name(resolved.name):
            raise AttachmentError(
                f"attachment {raw!r} is refused by name: dotfiles and files matching "
                f"{', '.join(DENYLIST_PATTERNS)} are never attachable."
            )

        if resolved.is_dir():
            raise AttachmentError(
                f"attachment {raw!r} is a directory; only regular files can be attached."
            )
        if not resolved.is_file():
            raise AttachmentError(
                f"attachment {raw!r} is not a regular file; only regular files can be "
                f"attached."
            )
        try:
            size = resolved.stat().st_size
        except OSError as e:
            raise AttachmentError(f"attachment {raw!r} could not be read: {e}") from None
        staged.append((raw, resolved, name, size))

    _check_total(staged, max_total_bytes)

    # Pass 2: read and decode.
    loaded: list[Attachment] = []
    actual: list[tuple[str, Path, str, int]] = []
    for raw, resolved, name, _size in staged:
        try:
            data = resolved.read_bytes()
        except OSError as e:
            raise AttachmentError(f"attachment {raw!r} could not be read: {e}") from None
        actual.append((raw, resolved, name, len(data)))
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise AttachmentError(
                f"attachment {raw!r} is not valid UTF-8 text (undecodable byte at "
                f"offset {e.start}); only UTF-8 text files can be attached in this "
                f"version — binary, PDF and image attachments are not supported."
            ) from None
        loaded.append(
            Attachment(
                name=name,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                content=text,
            )
        )
    # A file may have grown between stat() and read(); re-check what was read.
    _check_total(actual, max_total_bytes)
    return loaded


def _check_total(entries: list[tuple[str, Path, str, int]], cap: int) -> None:
    total = sum(size for _raw, _resolved, _name, size in entries)
    if total <= cap:
        return
    largest = max(entries, key=lambda e: e[3])
    raise AttachmentError(
        f"attachments total {total:,} bytes, over the {cap:,}-byte cap "
        f"(max_attachment_bytes); the largest is {largest[2]!r} at {largest[3]:,} "
        f"bytes. Attach fewer or smaller files, or raise max_attachment_bytes in "
        f"config.json / LLM_SECOND_OPINION_MAX_ATTACHMENT_BYTES."
    )


__all__ = [
    "Attachment",
    "AttachmentError",
    "DENYLIST_PATTERNS",
    "DIGEST_LOG_CHARS",
    "is_denied_name",
    "is_strictly_inside",
    "load_attachments",
    "paths_are_case_insensitive",
    "render_attachment",
    "resolve_roots",
]
