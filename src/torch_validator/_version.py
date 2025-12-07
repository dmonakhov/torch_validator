"""
Version info for torch_validator.

Git commit is read dynamically at runtime from the source directory.
"""

import subprocess
from pathlib import Path

__version__ = "0.1.0"

# Cached git info (computed once on first access)
_git_info_cache = None


def _get_git_info():
    """Get git commit and dirty status from source directory."""
    global _git_info_cache
    if _git_info_cache is not None:
        return _git_info_cache

    # Find the repo root (parent of src/)
    try:
        src_dir = Path(__file__).parent.parent.parent
        git_dir = src_dir / ".git"

        if not git_dir.exists():
            _git_info_cache = ("unknown", False)
            return _git_info_cache

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=src_dir,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=src_dir,
            capture_output=True,
            text=True
        )
        dirty = bool(result.stdout.strip())

        _git_info_cache = (commit, dirty)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        _git_info_cache = ("unknown", False)

    return _git_info_cache


def get_version_info() -> dict:
    """Get full version info as dict."""
    commit, dirty = _get_git_info()
    return {
        "version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def get_version_string() -> str:
    """Get version string with git info."""
    commit, dirty = _get_git_info()
    dirty_str = "-dirty" if dirty else ""
    if commit == "unknown":
        return __version__
    return f"{__version__}+{commit[:8]}{dirty_str}"


