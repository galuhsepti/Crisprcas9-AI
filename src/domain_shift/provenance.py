"""
Phase 14 provenance.

Records git HEAD/status, SHA-256 file hashes, the config/protocol hash, dataset
sizes, bin boundaries and software environment versions. Canonical model
hashes are recorded at FEASIBILITY and re-verified at AUDIT (must be identical).
"""

import hashlib
import json
import subprocess
import dataclasses
from pathlib import Path
from typing import Dict


def sha256_file(path) -> str:
    """SHA-256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_status(root: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=root
        )
        return out.stdout
    except Exception:
        return "unknown"


def config_hash(cfg) -> str:
    """Deterministic SHA-256 of the canonical serialized frozen config."""
    raw = json.dumps(
        dataclasses.asdict(cfg), sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def env_versions() -> Dict[str, str]:
    """Software environment versions available at runtime."""
    import sys

    out = {"python": sys.version.split()[0]}
    for pkg in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "pytest", "torch"]:
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as err:  # pragma: no cover
            out[pkg] = f"unavailable ({err})"
    return out


def bin_sizes(values, edges, rightmost_closed: bool = True) -> Dict[str, int]:
    """Counts per frozen bin for a dataset (provenance record)."""
    import numpy as np
    from .stratification import bin_indices

    idx = bin_indices(values, edges, rightmost_closed=rightmost_closed)
    n_bins = len(edges) + 1
    return {f"bin_{i}": int(np.sum(idx == i)) for i in range(n_bins)}