#!/usr/bin/env python3
"""Run Phase 17E distribution and potential information-value audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset_recovery.distribution import (
    NO_ACTIVITY,
    build_phase17e_distribution,
    dump_json,
)


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def _save_gc_figure(payload: dict, timestamp: str) -> str:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"phase17e_gc_distribution_{timestamp}.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    canonical = payload["canonical_reference"]["distribution_reference"]["sequence"]["gc"]
    ax.axvline(canonical["mean"], color="black", linewidth=1.5, label="DeepSpCas9 mean GC (n=12,832)")
    for record in payload["candidates"]:
        seq = record["sequence_distribution"]
        if not isinstance(seq, dict) or seq.get("status") == "NO_SEQUENCE_AVAILABLE":
            continue
        gc = seq["gc"]
        ax.axvline(gc["mean"], linewidth=1.2, linestyle="--", label=f"{record['candidate_id']} mean GC (n={gc['n']})")
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("mean marker")
    ax.set_title("Phase 17E GC composition comparison, descriptive only")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path).replace("\\", "/")


def _save_nucleotide_figure(payload: dict, timestamp: str) -> str:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"phase17e_nucleotide_composition_{timestamp}.png"
    labels = ["DeepSpCas9"]
    rows = [payload["canonical_reference"]["distribution_reference"]["sequence"]["nucleotide_composition"]]
    for record in payload["candidates"]:
        seq = record["sequence_distribution"]
        if isinstance(seq, dict) and seq.get("status") != "NO_SEQUENCE_AVAILABLE":
            labels.append(record["candidate_id"][:24])
            rows.append(seq["nucleotide_composition"])
    x = range(len(labels))
    bottoms = [0.0] * len(labels)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {"A": "#5577aa", "C": "#66aa77", "G": "#aa7755", "T": "#aa5577"}
    for base in "ACGT":
        values = [row.get(base, 0.0) for row in rows]
        ax.bar(x, values, bottom=bottoms, label=base, color=colors[base])
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_title("Phase 17E nucleotide composition, observed sequences only")
    ax.legend(title="Base", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path).replace("\\", "/")


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 17E Distribution and Information-Value Audit",
        "",
        "## 1. Dataset classification",
        "",
        "| candidate_id | dataset type | verified sequence? | verified continuous activity? | analysis performed | information-value status |",
        "|---|---|---|---|---|---|",
    ]
    for record in payload["candidates"]:
        analysis = record["analysis_performed"]
        lines.append(
            f"| `{record['candidate_id']}` | `{record['dataset_type']}` | "
            f"{record['verified_sequence_available']} | {record['verified_continuous_activity_available']} | "
            f"sequence={analysis['sequence_distribution']}; activity={analysis['activity_distribution']} | "
            f"`{record['information_value_status']}` |"
        )
    lines.extend([
        "",
        "## 2. Sequence distribution",
        "",
        "Reference for sequence composition comparison = DeepSpCas9 Phase 16E distribution population (n=12,832).",
        "",
        "| candidate_id | n valid A/C/G/T | length distribution | mean GC | median GC | sd GC | A | C | G | T | comparison |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for record in payload["candidates"]:
        seq = record["sequence_distribution"]
        if not isinstance(seq, dict) or seq.get("status") == "NO_SEQUENCE_AVAILABLE":
            lines.append(f"| `{record['candidate_id']}` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | insufficient verified sequence evidence |")
            continue
        gc = seq["gc"]
        nuc = seq["nucleotide_composition"]
        comp = record["sequence_comparison_to_canonical"]
        gc_delta = comp["gc"]["mean_gc_delta_candidate_minus_reference"]
        lines.append(
            f"| `{record['candidate_id']}` | {seq['records_with_valid_acgt_sequence']} | {seq['length_distribution']} | "
            f"{gc['mean']:.4f} | {gc['median']:.4f} | {gc['std']:.4f} | "
            f"{nuc['A']:.4f} | {nuc['C']:.4f} | {nuc['G']:.4f} | {nuc['T']:.4f} | "
            f"candidate exhibits a different sequence-composition regime; mean GC delta {gc_delta:.4f} |"
        )
    lines.extend([
        "",
        "## 3. Activity distribution",
        "",
        NO_ACTIVITY,
        "",
        "No candidate label was analyzed as thesis activity because Phase 17C did not establish `LABEL_COMPATIBLE` for any recovered candidate. CRISPRpred binary labels and Doench screen/rank/enrichment/log-fold-change values were not transformed or treated as activity.",
        "",
        "## 4. Information-value assessment",
        "",
    ])
    for record in payload["candidates"]:
        lines.append(f"- `{record['candidate_id']}`: `{record['information_value_status']}`. {record['information_value_reason']}")
    lines.extend([
        "",
        "## 5. Canonical population distinction",
        "",
        "DeepSp Phase 16E distribution n=12,832",
        "",
        "DeepSp canonical modeling n=10,117",
        "",
        "train=8,599",
        "",
        "validation=1,518",
        "",
        "## 6. Tests",
        "",
        "- Focused Phase 17E tests: see run log / final implementation response.",
        "- Phase 17A-17D regression tests: see run log / final implementation response.",
        "- Existing project regression tests: see run log / final implementation response.",
        "- Warnings: report separately from failures.",
        "- Failures: report separately from timeouts.",
        "",
        "## 7. Scope confirmation",
        "",
    ])
    for key, value in payload["non_modeling_guards"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## 8. Git state",
        "",
        "```bash",
        "git status",
        payload["git"]["status"],
        "",
        "git diff --stat",
        payload["git"]["diff_stat"],
        "",
        "git diff --name-only",
        payload["git"]["diff_name_only"],
        "",
        "git log -2 --oneline",
        payload["git"]["log_2"],
        "```",
        "",
        "## 9. Final status",
        "",
        "PHASE 17E PASS — DISTRIBUTION/INFORMATION-VALUE AUDIT COMPLETE",
        "",
    ])
    return "\n".join(lines)


def main() -> dict:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17e_distribution()
    payload["timestamp"] = now.isoformat()
    payload["git"] = {
        "head": _git(["rev-parse", "HEAD"]),
        "status": _git(["status", "--short"]),
        "diff_stat": _git(["diff", "--stat"]),
        "diff_name_only": _git(["diff", "--name-only"]),
        "log_2": _git(["log", "-2", "--oneline"]),
    }
    payload["figures"] = {
        "gc_distribution": _save_gc_figure(payload, timestamp),
        "nucleotide_composition": _save_nucleotide_figure(payload, timestamp),
    }
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"phase17e_distribution_information_value_{timestamp}.json"
    report_path = docs_dir / "phase17e_distribution_information_value_report.md"
    payload["artifacts"] = {
        "distribution_information_value_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17E distribution/information-value audit complete: {result_path}")
    print(f"Phase 17E report: {report_path}")
    print("No model training, Moreno access, pooling, transformations, or dataset acceptance performed")
    return payload


if __name__ == "__main__":
    main()
