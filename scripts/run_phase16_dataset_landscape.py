#!/usr/bin/env python3
"""Phase 16A - public candidate discovery/provenance inventory only."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_landscape.discovery import build_phase16a_inventory
from src.dataset_landscape.reporting import render_phase16a_report


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def git_status() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "(clean)"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def main() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    access_date = datetime.now().date().isoformat()
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_phase16a_inventory(access_date=access_date)
    inventory["timestamp"] = datetime.now().isoformat()
    inventory["git"] = {
        "head": git_head(),
        "status_before_output": git_status(),
    }
    inventory["artifacts"] = {
        "inventory_json": str(results_dir / f"phase16a_dataset_inventory_{timestamp}.json"),
        "provenance_json": str(results_dir / f"phase16a_provenance_audit_{timestamp}.json"),
        "report_md": str(docs_dir / "phase16a_dataset_landscape_report.md"),
    }

    inventory_path = Path(inventory["artifacts"]["inventory_json"])
    provenance_path = Path(inventory["artifacts"]["provenance_json"])
    report_path = Path(inventory["artifacts"]["report_md"])

    inventory_path.write_text(json.dumps(inventory, indent=2, default=str) + "\n")
    provenance_payload = {
        "phase": inventory["phase"],
        "scope": inventory["scope"],
        "status": inventory["status"],
        "access_date": inventory["access_date"],
        "discovery_sources_consulted": inventory["discovery_sources_consulted"],
        "provenance_validation": inventory["provenance_validation"],
        "candidates": inventory["candidates"],
        "non_modeling_guards": inventory["non_modeling_guards"],
    }
    provenance_path.write_text(json.dumps(provenance_payload, indent=2, default=str) + "\n")
    report_path.write_text(render_phase16a_report(inventory))

    print("Phase 16A discovery/provenance inventory complete")
    print(f"Inventory: {inventory_path}")
    print(f"Provenance: {provenance_path}")
    print(f"Report: {report_path}")
    print("No Moreno raw data accessed")
    print("No model training/evaluation performed")
    return inventory


if __name__ == "__main__":
    main()
