#!/usr/bin/env python3
"""Run the audit-only Phase 18A label compatibility assessment."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_integration.label_compatibility import build_phase18a_audit


def _format(value: object, digits: int = 6) -> str:
    if value is None:
        return "NOT_ESTABLISHED"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(payload: dict) -> str:
    semantics = payload["label_semantics"]
    distributions = payload["label_distribution_comparison"]
    bridge = payload["shared_sequence_bridge"]
    biology = payload["biological_domain_comparison"]
    sequence = payload["sequence_domain_comparison"]
    lines = [
        "# Phase 18A - Cross-Dataset Label Compatibility and Integration Strategy Audit",
        "",
        "## Scope",
        "",
        "- Audit only; no model training, retraining, tuning, calibration, or evaluation occurred.",
        "- No label normalization, transformation, harmonization fit, or training dataset creation occurred.",
        "- The locked external dataset was not accessed.",
        "- Canonical model code, data, artifacts, and evaluation policy remain unchanged.",
        "",
        "## Primary Answer",
        "",
        f"- Immediate direct pooling justified: **{payload['immediate_pooling_justified']}**",
        f"- Controlled integration experiments justified: **{payload['controlled_integration_experiments_justified']}**",
        f"- Primary recommendation: **{payload['recommended_strategy']}**",
        "",
        "Phase 17G-R1 GO establishes data sufficiency for controlled experiments. It does not establish that the two response variables are directly exchangeable.",
        "",
        "## Label Semantics",
        "",
    ]
    for dataset_id in ("deepspcas9", "crispron_xiang_luo"):
        item = semantics[dataset_id]
        lines.extend(
            [
                f"### {dataset_id}",
                "",
                f"- Publication: {item['original_publication']} ({item['doi']})",
                f"- Experimental assay: {item['experimental_assay']}",
                f"- Cell type(s): {', '.join(item['cell_types'])}",
                f"- Target system: {item['target_system']}",
                f"- Cas nuclease: {item['cas_nuclease']}",
                f"- Label: `{item['label_name']}`",
                f"- Activity definition: {item['activity_definition']}",
                f"- Raw biological measurement: {item['raw_biological_measurement']}",
                f"- Normalization: {item['normalization']}",
                f"- Numerical unit: {item['numerical_unit']}",
                f"- Higher means greater editing efficiency: {item['higher_means_greater_editing_efficiency']}",
                f"- Direct primary measurement: {item['direct_primary_measurement']}",
                f"- Original-author transformations: {item['original_author_transformations']}",
                f"- Timing: {item['measurement_timing']}",
                f"- Evidence caveat: {item['evidence_caveat']}",
                "- Evidence chain:",
            ]
        )
        lines.extend(
            [
                f"  {index}. {entry}"
                for index, entry in enumerate(item["evidence_chain"], 1)
            ]
        )
        lines.append("")
    concept = semantics["underlying_concept_assessment"]
    lines.extend(
        [
            "### Concept Comparison",
            "",
            f"- Same underlying biological concept: **{concept['same_underlying_biological_concept']}**",
            f"- Concept: {concept['concept']}",
            f"- Operationally equivalent endpoints: **{concept['operationally_equivalent_endpoints']}**",
            f"- Reason: {concept['reason']}",
            "",
            "## Label Scale Comparison",
            "",
            f"- Numerical scales directly comparable: **{distributions['numerical_scales_directly_comparable']}**",
            "- No values were converted, normalized, ranked, or otherwise transformed.",
            "",
            "| statistic | DeepSpCas9 canonical n=10,117 | Xiang/Luo n=10,592 |",
            "|---|---:|---:|",
        ]
    )
    deep_stats = distributions["deepspcas9_canonical_modeling_population"]
    xiang_stats = distributions["crispron_xiang_luo"]
    for key in (
        "n",
        "finite_n",
        "missing_n",
        "min",
        "max",
        "mean",
        "median",
        "SD",
        "IQR",
        "unique_value_n",
        "skewness",
        "fraction_at_min",
        "fraction_at_max",
    ):
        lines.append(
            f"| {key} | {_format(deep_stats[key])} | {_format(xiang_stats[key])} |"
        )
    lines.extend(
        [
            "",
            "Quantiles:",
            "",
            "| percentile | DeepSpCas9 | Xiang/Luo |",
            "|---:|---:|---:|",
        ]
    )
    for percentile in ("0", "1", "5", "10", "25", "50", "75", "90", "95", "99", "100"):
        lines.append(
            f"| {percentile} | {_format(deep_stats['quantiles_percent'][percentile])} | {_format(xiang_stats['quantiles_percent'][percentile])} |"
        )
    lines.extend(
        [
            "",
            distributions["interpretation"],
            "",
            "## Shared-Sequence Bridge",
            "",
            f"- DeepSpCas9 bridge population: **{bridge['deepspcas9_population']['name']} (n={bridge['deepspcas9_population']['n']})**",
            f"- Exact shared 30-mers: **{bridge['n']}**",
            f"- Finite label pairs: {bridge['finite_pair_n']}",
            f"- Pearson r: {_format(bridge['pearson']['coefficient'])} (p={_format(bridge['pearson']['p_value'])})",
            f"- Spearman rho: {_format(bridge['spearman']['coefficient'])} (p={_format(bridge['spearman']['p_value'])})",
            f"- Kendall tau: {_format(bridge['kendall']['coefficient'])} (p={_format(bridge['kendall']['p_value'])})",
            f"- Monotonic relationship: {bridge['monotonic_relationship']}",
            f"- Mean absolute percentile-rank disagreement: {_format(bridge['rank_agreement']['mean_absolute_percentile_rank_disagreement'])}",
            f"- Same-quartile fraction: {_format(bridge['rank_agreement']['same_quartile_fraction'])}",
            f"- Linear slope/intercept (diagnostic only): {_format(bridge['linear_regression'].get('slope'))} / {_format(bridge['linear_regression'].get('intercept'))}",
            f"- Theil-Sen slope/intercept (diagnostic only): {_format(bridge['robust_regression'].get('slope'))} / {_format(bridge['robust_regression'].get('intercept'))}",
            f"- Regression outliers: {bridge['outlier_n']}",
            "",
            "Raw numeric disagreement is reported in JSON but is not interpreted because one source uses fractions and the other percentages. The bridge was not used to fit a future-training harmonization transform.",
            "",
            "### Canonical-Population Sensitivity",
            "",
            f"- DeepSpCas9 canonical population: n={bridge['canonical_modeling_population_sensitivity']['deepspcas9_n']}",
            f"- Exact shared 30-mers: {bridge['canonical_modeling_population_sensitivity']['shared_30mer_n']}",
            f"- Pearson r: {_format(bridge['canonical_modeling_population_sensitivity']['pearson']['coefficient'])}",
            f"- Spearman rho: {_format(bridge['canonical_modeling_population_sensitivity']['spearman']['coefficient'])}",
            f"- Relationship: {bridge['canonical_modeling_population_sensitivity']['monotonic_relationship']}",
            f"- Interpretation: {bridge['canonical_modeling_population_sensitivity']['interpretation']}",
            "",
            "| 30-mer | DeepSpCas9 | Xiang/Luo | Deep rank | Xiang rank | abs rank disagreement | outlier |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in bridge["rows"]:
        lines.append(
            f"| `{row['sequence_30mer']}` | {_format(row['deepspcas9_label'])} | {_format(row['crispron_xiang_luo_label'])} | "
            f"{_format(row['deepspcas9_percentile_rank'])} | {_format(row['crispron_percentile_rank'])} | "
            f"{_format(row['absolute_percentile_rank_disagreement'])} | {row['linear_outlier']} |"
        )
    lines.extend(
        [
            "",
            "## Biological Domains",
            "",
            "| factor | DeepSpCas9 | Xiang/Luo | severity |",
            "|---|---|---|---|",
        ]
    )
    for item in biology["differences"]:
        lines.append(
            f"| {item['factor']} | {item['deepspcas9']} | {item['crispron_xiang_luo']} | {item['classification']} |"
        )
    lines.extend(
        [
            "",
            f"Major differences: {', '.join(biology['major_domain_differences'])}.",
            "",
            "Unknown metadata:",
        ]
    )
    lines.extend([f"- {item}" for item in biology["unknown_metadata"]])
    comparison = sequence["comparison"]
    lines.extend(
        [
            "",
            "## Sequence Domains",
            "",
            f"- DeepSpCas9 population: canonical modeling n={sequence['deepspcas9']['n']}",
            f"- Xiang/Luo population: n={sequence['crispron_xiang_luo']['n']}",
            f"- Mean 30-mer GC delta (Xiang/Luo minus DeepSpCas9): {_format(comparison['full_30mer_gc_mean_delta_xiang_minus_deep'])}",
            f"- Mean spacer GC delta: {_format(comparison['spacer_gc_mean_delta_xiang_minus_deep'])}",
            f"- PAM JSD (bits): {_format(comparison['PAM_JSD_bits'])}",
            f"- Mean/max positional nucleotide JSD (bits): {_format(comparison['mean_positional_JSD_bits'])} / {_format(comparison['max_positional_JSD_bits'])}",
            f"- Full 30-mer k=2/k=3 JSD: {_format(comparison['kmer']['full_30mer']['k2']['JSD_bits'])} / {_format(comparison['kmer']['full_30mer']['k3']['JSD_bits'])}",
            f"- Spacer k=2/k=3 JSD: {_format(comparison['kmer']['spacer20']['k2']['JSD_bits'])} / {_format(comparison['kmer']['spacer20']['k3']['JSD_bits'])}",
            f"- Interpretation: {comparison['interpretation']}",
            "",
            "Full positional frequencies, PAM distributions, k-mer summaries, entropy, and complexity descriptors are retained in the JSON artifact.",
            "",
            "## Integration Strategies",
            "",
            "| strategy | classification | central assessment |",
            "|---|---|---|",
        ]
    )
    labels = {
        "direct_pooling": "A DIRECT_POOLING",
        "normalization_then_pooling": "B WITHIN-DATASET NORMALIZATION THEN POOLING",
        "rank_based": "C RANK-BASED INTEGRATION",
        "multi_domain": "D MULTI-DOMAIN LEARNING",
        "transfer_learning": "E TRANSFER LEARNING",
        "external_development_domain": "F EXTERNAL DEVELOPMENT DOMAIN",
    }
    for key, label in labels.items():
        strategy = payload["integration_strategies"][key]
        lines.append(
            f"| {label} | **{strategy['classification']}** | {strategy['scientific_defensibility']} |"
        )
    lines.extend(
        [
            "",
            "Detailed assumptions and risks for each strategy are in the JSON artifact.",
            "",
            "## Decision Reasons",
            "",
        ]
    )
    lines.extend([f"- {reason}" for reason in payload["decision_reasons"]])
    integrity = payload["canonical_integrity"]
    lines.extend(
        [
            "",
            "## Canonical Integrity",
            "",
            f"- Safe-file hashes unchanged during audit: {integrity['unchanged']}",
            f"- All expected safe-file hashes match: {integrity['all_expected_hashes_match']}",
            f"- Canonical data modified: {integrity['canonical_data_modified']}",
            f"- Canonical model code modified: {integrity['canonical_model_code_modified']}",
            f"- Canonical artifacts modified: {integrity['canonical_artifacts_modified']}",
            f"- Locked external data accessed: {integrity['locked_external_data_accessed']}",
            "",
            "## Stop",
            "",
            "Phase 18B was not started. Approval is required before any integration experiment or label transformation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase18a_audit(timestamp=now.isoformat())
    result_path = Path("results") / f"phase18a_cross_dataset_compatibility_{stamp}.json"
    report_path = Path("docs") / "phase18a_cross_dataset_compatibility_report.md"
    payload["artifacts"] = {
        "json": str(result_path).replace("\\", "/"),
        "report": str(report_path).replace("\\", "/"),
    }
    result_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_report(payload) + "\n", encoding="utf-8")
    print(f"Phase 18A JSON: {result_path}")
    print(f"Phase 18A report: {report_path}")
    print(f"Recommendation: {payload['recommended_strategy']}")
    return payload


if __name__ == "__main__":
    main()
