#!/usr/bin/env python3
"""Phase 16E - conservative distribution/composition audit only."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset_landscape.distribution import build_phase16e_distribution_audit


def _git(args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def save_gc_figure(payload: dict, timestamp: str) -> str:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"phase16e_gc_distribution_{timestamp}.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    canon_hist = payload["canonical_reference"]["sequence"]["gc_histogram"]
    ax.stairs(canon_hist["counts_density"], canon_hist["bin_edges"], label="DeepSpCas9 30-mer")
    for record in payload["candidate_reports"]:
        seq = record["sequence"]
        if not isinstance(seq, dict) or "gc_histogram" not in seq:
            continue
        hist = seq["gc_histogram"]
        ax.stairs(hist["counts_density"], hist["bin_edges"], label=record["candidate_id"][:24], alpha=0.7)
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("count")
    ax.set_title("Phase 16E observed-sequence GC distributions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 16E Distribution / Composition Audit",
        "",
        "**Status:** " + payload["status"],
        "**Scope:** Descriptive distribution and composition audit only.",
        "",
        "No modeling, label transformation, pooling, overlap reinterpretation, "
        "gate decision, or Moreno access was performed.",
        "",
        "## Dataset Counts",
        "",
        "| dataset | sequence records | compatibility | activity analyzed |",
        "|---|---:|---|---|",
        f"| DeepSpCas9 | {payload['canonical_reference']['record_counts']['raw_records']} | canonical reference | yes |",
    ]
    for record in payload["candidate_reports"]:
        lines.append(
            f"| {record['candidate_id']} | {record['record_counts']['sequence_records']} | "
            f"{record['compatibility_status_from_16c']} | no |"
        )
    lines.extend(["", "## Sequence GC Summary", "", "| dataset | n | mean | median | sd |"])
    lines.append("|---|---:|---:|---:|---:|")
    cgc = payload["canonical_reference"]["sequence"]["gc"]["descriptive"] if "descriptive" in payload["canonical_reference"]["sequence"].get("gc", {}) else payload["canonical_reference"]["sequence"]["gc"]
    lines.append(f"| DeepSpCas9 | {cgc['n']} | {cgc['mean']:.4f} | {cgc['median']:.4f} | {cgc['std']:.4f} |")
    for record in payload["candidate_reports"]:
        seq = record["sequence"]
        if not isinstance(seq, dict) or seq.get("status") == "NO_SEQUENCE_AVAILABLE":
            lines.append(f"| {record['candidate_id']} | 0 | n/a | n/a | n/a |")
            continue
        gc = seq["gc"]
        lines.append(f"| {record['candidate_id']} | {gc['n']} | {gc['mean']:.4f} | {gc['median']:.4f} | {gc['std']:.4f} |")
    lines.extend([
        "",
        "## Activity / Label Distribution",
        "",
        "- DeepSpCas9 activity was analyzed as the canonical established continuous label.",
        "- Candidate labels were not analyzed as activity because Phase 16C did not establish target-compatible continuous activity.",
        "",
        "## Phase 16D Context Carried Forward",
        "",
    ])
    for record in payload["candidate_reports"]:
        ctx = record["overlap_context_from_16d"]
        lines.append(f"- `{record['candidate_id']}` RC: `{ctx['rc_overlap_status']['status']}`; independence: `{record['provenance_independence_from_16d']['independence_assessment']}`")
    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for name, path in payload.get("figures", {}).items():
        lines.append(f"- {name}: `{path}`")
    lines.extend([
        "",
        "## Locks",
        "",
        "- No Moreno raw data accessed.",
        "- No model training/evaluation performed.",
        "- No dataset pooling performed.",
        "- No Phase 16F gate decision performed.",
        "- No canonical Phase 3-15 artifacts modified.",
    ])
    return "\n".join(lines) + "\n"


def main() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = build_phase16e_distribution_audit(datetime.now().date().isoformat())
    payload["timestamp"] = datetime.now().isoformat()
    payload["git"] = {"head": _git(["rev-parse", "HEAD"]), "status_before_output": _git(["status", "--porcelain"])}
    payload["figures"] = {"gc_distribution": save_gc_figure(payload, timestamp)}
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"phase16e_distribution_audit_{timestamp}.json"
    report_path = docs_dir / "phase16e_distribution_audit_report.md"
    payload["artifacts"] = {"distribution_json": str(result_path), "report_md": str(report_path)}
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    report_path.write_text(render_report(payload))
    print("Phase 16E distribution audit complete")
    print(f"Distribution: {result_path}")
    print(f"Report: {report_path}")
    print(f"Figure: {payload['figures']['gc_distribution']}")
    print("No Moreno raw data accessed")
    print("No model training/evaluation performed")
    print("No dataset pooling performed")
    return payload


if __name__ == "__main__":
    main()
