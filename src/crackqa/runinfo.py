"""Provenance recorded with every result: git commit, dirty files, hardware."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import torch


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def git_commit(cwd: Path = Path(".")) -> str | None:
    result = _git(["rev-parse", "HEAD"], cwd)
    return result.stdout.strip() if result.returncode == 0 else None


def git_problems(paths: list[Path]) -> list[str]:
    """Each path must be tracked by Git and have no uncommitted change."""
    problems = []
    for path in paths:
        cwd, name = path.resolve().parent, path.name
        if _git(["ls-files", "--error-unmatch", name], cwd).returncode != 0:
            problems.append(f"{path.as_posix()} is not committed to Git")
        elif _git(["status", "--porcelain", "--", name], cwd).stdout.strip():
            problems.append(f"{path.as_posix()} has uncommitted changes")
    return problems


def hardware() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "device": "cpu",
    }
