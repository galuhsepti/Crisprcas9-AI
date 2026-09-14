"""Training-compatible filesystem guard for the Phase 18C locked dataset."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
LOCKED_RELATIVE_PATH = Path("data/raw/Moreno-Mateos.csv")


def _portable(path: Path) -> str:
    return str(path).replace("\\", "/")


def _reject_locked_name(path: Path) -> None:
    text = _portable(path).casefold()
    if "moreno" in text or "mateos" in text:
        raise PermissionError("Phase 18C locked external dataset access prohibited")


def _resolve_path(path: Path) -> Path:
    return path.resolve()


def resolve_phase18c_path(path: str | bytes | Path) -> Path:
    """Resolve an allowed path while rejecting locked names and aliases."""
    candidate = Path(os.fsdecode(path))
    _reject_locked_name(candidate)
    resolved = _resolve_path(candidate)
    _reject_locked_name(resolved)
    locked = (ROOT / LOCKED_RELATIVE_PATH).absolute()
    if resolved == locked:
        raise PermissionError("Phase 18C locked external dataset access prohibited")
    return resolved


@contextmanager
def phase18c_access_guard() -> Iterator[None]:
    """Reject locked file opens without restricting registered model training."""
    state = {"active": True}

    def on_audit(event: str, args: tuple[object, ...]) -> None:
        if not state["active"] or event != "open" or not args:
            return
        name = args[0]
        if isinstance(name, (str, bytes, Path)):
            resolve_phase18c_path(name)

    sys.addaudithook(on_audit)
    try:
        yield
    finally:
        state["active"] = False
