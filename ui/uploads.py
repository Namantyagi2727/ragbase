"""
ui/uploads.py — Per-session workspaces for uploaded documents and their index (local mode).

Each browser session gets its own temporary directory for uploads and the index
built from them, so concurrent sessions never share files. Workspaces are
removed when a session builds a new one and, best effort, when they have been
idle for longer than a retention period (checked at start-up of new sessions). Streamlit has no
reliable end-of-session hook, so a workspace can outlive its session until
then. The pre-existing shared folders (config.DOCS_DIR / config.INDEX_DIR) are
never modified or deleted here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PREFIX = "ragbase-session-"
ROOT = Path(os.environ.get("RAGBASE_SESSION_ROOT", tempfile.gettempdir())) / "ragbase-sessions"
RETENTION_S = float(os.environ.get("RAGBASE_SESSION_RETENTION_S", 6 * 3600))


@dataclass
class Workspace:
    path: Path

    @property
    def docs_dir(self) -> str:
        return str(self.path / "docs")

    @property
    def index_dir(self) -> str:
        return str(self.path / "index")


def create(root: Path | None = None) -> Workspace:
    root = root or ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=PREFIX, dir=root))
    (path / "docs").mkdir()
    return Workspace(path)


def remove(ws: Workspace | None, root: Path | None = None) -> None:
    """Delete a workspace created by create(); refuses anything outside the session root."""
    if ws is None:
        return
    root = root or ROOT
    path = ws.path.resolve()
    if path.parent != root.resolve() or not path.name.startswith(PREFIX):
        raise ValueError(f"refusing to delete {path}: not a RAGBase session workspace")
    shutil.rmtree(path, ignore_errors=True)


def touch(ws: Workspace) -> None:
    """Mark a workspace as in use (its age for sweep() restarts)."""
    try:
        os.utime(ws.path)
    except FileNotFoundError:
        pass


def sweep(root: Path | None = None, retention_s: float = RETENTION_S, now: float | None = None) -> int:
    """Remove session workspaces idle for longer than the retention period. Returns how many were removed."""
    root = root or ROOT
    if not root.exists():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(PREFIX) and now - child.stat().st_mtime > retention_s:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
