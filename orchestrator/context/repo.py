"""Repo state probe — git status for configured repositories."""

from __future__ import annotations

import logging
from pathlib import Path

from orchestrator.config import get_settings
from orchestrator.context.base import ContextBlock, estimate_tokens

log = logging.getLogger(__name__)


def _git_cmd(repo: Path, *args: str) -> str:
    # safe_run enforces the whitelist and timeout; cwd is not needed because
    # git accepts --git-dir / -C flags, but here we pass absolute paths via
    # the caller, so we temporarily chdir only via the subprocess cwd trick.
    # Instead, we use a thin wrapper that sets cwd.
    cfg = get_settings().security
    if "git" not in cfg.allowed_commands:
        return ""
    import subprocess  # local import — cwd support not available via safe_run
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True,
            cwd=str(repo), timeout=cfg.max_command_timeout,
            shell=False,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


class RepoProbeProvider:
    """Collects git state for configured repos."""

    @property
    def name(self) -> str:
        return "repo"

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        cfg = get_settings()
        repos = cfg.repos.paths
        if not repos:
            return None

        lines = ["# Repository State\n"]
        for repo_path in repos:
            repo_path = Path(repo_path)
            name = repo_path.name
            if not repo_path.is_dir():
                lines.append(f"## {name}\n- **Status:** not found\n")
                continue

            lines.append(f"## {name}")
            branch = _git_cmd(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
            if branch:
                lines.append(f"- **Branch:** {branch}")

            last_commit = _git_cmd(repo_path, "log", "-1", "--format=%h %s (%cr)")
            if last_commit:
                lines.append(f"- **Last commit:** {last_commit}")

            status = _git_cmd(repo_path, "status", "--porcelain", "--short")
            if status:
                changed = len(status.strip().splitlines())
                lines.append(f"- **Uncommitted changes:** {changed} files")
            else:
                lines.append("- **Working tree:** clean")
            lines.append("")

        content = "\n".join(lines)
        if not content.strip():
            return None

        return ContextBlock(
            source="repo",
            content=content,
            token_estimate=estimate_tokens(content),
        )

    def health(self) -> bool:
        return True
