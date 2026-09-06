"""
Phase 12 - Controlled Data-Coverage Ablation: matching + feasibility audit.

All matching is defined on TRAINING-DOMAIN information only. Nothing in this
module reads external data, and no subset here is ever chosen by model or
external performance.

The audience for the feasibility audit is the scientific question of Section 4:
can the available data support a controlled 5-arm isolation (A..E)? Matching
quality is reported per dimension (sample size, activity, sequence, domain),
never collapsed into a single invented score.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..audit.audit import DEFAULT_ACTIVITY_EDGES
from ..diagnostics.stats import cohens_d, ks_test
from ..diagnostics.sequence_analysis import (
    duplicate_summary,
    gc_content_array,
    mean_kmer_frequencies,
    nucleotide_frequency_array,
    sequence_length_summary,
    exact_sequence_overlap,
)
from ..dataset_analysis.analysis import (
    activity_descriptive,
    coverage_by_bin,
    js_divergence,
    sequence_diversity_report,
    wasserstein_1d,
)
from .config import (
    ACTIVITY_EDGES,
    ACTIVITY_QUANTILES,
    DECISION_GATES,
    PROPOSED_TOLERANCES,
    STOP_CONDITIONS,
)

# ---------------------------------------------------------------------------
# sample size
# ---------------------------------------------------------------------------


def sample_size_report(n_a: int, n_b: int) -> Dict:
    """Absolute and percentage sample-count difference."""
    pct = float((n_b - n_a) / n_a * 100.0) if n_a else None
    return {
        'n_a': int(n_a),
        'n_b': int(n_b),
        'abs_diff': int(n_b - n_a),
        'rel_pct': (round(pct, 3) if pct is not None else None),
        'within_proposed_tolerance': bool(
            pct is not None and abs(pct) <=
            PROPOSED_TOLERANCES['sample_size_rel_pct']),
    }


# ---------------------------------------------------------------------------
# activity distribution
# ---------------------------------------------------------------------------


def quantile_profile(values: Sequence[float],
                     qs: Sequence[int] = ACTIVITY_QUANTILES) -> Dict:
    """Named percentile summary of a label sample."""
    v = np.asarray(values, dtype=float).ravel()
    return {f'p{q}': float(np.percentile(v, q)) for q in qs}


def bin_fractions(values: Sequence[float],
                  edges: Sequence[float] = ACTIVITY_EDGES) -> Dict:
    """Fraction of rows in each canonical activity bin (see Phase 11 grid)."""
    return {b['bin']: b['fraction'] for b in coverage_by_bin(values, edges)}


def activity_matching_report(labels_a: Sequence[float],
                             labels_b: Sequence[float]) -> Dict:
    """
    Per-dimension activity-distribution comparison. Matches more than the
    mean: full quantile profile, canonical-bin profile, and a defensible
    distribution distance (1-D Wasserstein) plus Cohen's d.
    """
    a = np.asarray(labels_a, dtype=float).ravel()
    b = np.asarray(labels_b, dtype=float).ravel()
    da = activity_descriptive(a)
    db = activity_descriptive(b)
    fa = bin_fractions(a)
    fb = bin_fractions(b)
    max_bin_delta = max(abs(fb[k] - fa[k]) for k in fa)
    return {
        'n_a': int(a.size), 'n_b': int(b.size),
        'mean': {'a': float(a.mean()), 'b': float(b.mean()),
                 'delta': float(b.mean() - a.mean())},
        'std': {'a': float(a.std(ddof=1)), 'b': float(b.std(ddof=1)),
                'delta': float(b.std(ddof=1) - a.std(ddof=1))},
        'quantiles': {
            'a': quantile_profile(a), 'b': quantile_profile(b),
            'max_abs_quantile_delta': float(max(
                abs(quantile_profile(b)[k] - quantile_profile(a)[k])
                for k in quantile_profile(a)))},
        'bins': {
            'a': fa, 'b': fb, 'max_abs_bin_delta': float(max_bin_delta)},
        'fraction_below_0.2': {'a': fa['[0.00, 0.20]'],
                               'b': fb['[0.00, 0.20]']},
        'wasserstein_1d': wasserstein_1d(a, b),
        'cohens_d': cohens_d(a, b),
        'ks_test': ks_test(a, b),
        'high_activity_>0.8': {
            'a': da['fraction_high_>0.8'], 'b': db['fraction_high_>0.8']},
    }


# ---------------------------------------------------------------------------
# sequence/domain distribution
# ---------------------------------------------------------------------------


def sequence_matching_report(seqs_a: Sequence[str],
                             seqs_b: Sequence[str]) -> Dict:
    """
    Composition/diversity comparison between two sequence sets under the
    canonical 30-mer geometry: GC, nucleotide composition, 2-/3-mer JSD,
    uniqueness and duplicate rate. A GC difference alone is not biological
    diversity (interpretation is left to the report).
    """
    a = list(seqs_a)
    b = list(seqs_b)
    div = sequence_diversity_report(a, b, add_name='b')
    du_a = duplicate_summary(a)
    du_b = duplicate_summary(b)
    return {
        'gc': div['gc'],
        'nucleotide': div['nucleotide_composition'],
        'kmer_2': div['kmer_2'],
        'kmer_3': div['kmer_3'],
        'unique_sequences': {'a': int(du_a['unique']),
                             'b': int(du_b['unique'])},
        'duplicate_rate': {'a': float(du_a['duplicate_rate']),
                           'b': float(du_b['duplicate_rate'])},
        'length': {'a': sequence_length_summary(a),
                   'b': sequence_length_summary(b)},
    }


def domain_matching_report(frames: Dict[str, pd.DataFrame],
                           sources_a: Sequence[str],
                           sources_b: Sequence[str]) -> Dict:
    """
    Source/experimental-context comparison: source proportions, unique source
    count, context identity. If source and sequence composition both differ,
    the caller must state that the two dimensions remain confounded.
    """
    def _props(srcs: Sequence[str]) -> Dict[str, float]:
        tot = len(srcs)
        return {str(s): round(float(sum(1 for x in srcs if x == s)) / tot, 6)
                for s in sorted(set(srcs))}

    return {
        'sources_a': sorted(set(sources_a)),
        'sources_b': sorted(set(sources_b)),
        'n_unique_sources_a': int(len(set(sources_a))),
        'n_unique_sources_b': int(len(set(sources_b))),
        'source_proportions_a': _props(sources_a),
        'source_proportions_b': _props(sources_b),
        'shared_sources': sorted(set(sources_a) & set(sources_b)),
        'exact_30mer_overlap': exact_sequence_overlap(
            list(frames['a']['normalized_sequence']),
            list(frames['b']['normalized_sequence'])),
        'note': ('Source proportions are reported per arm; identity of '
                 'experimental context is tracked as text provenance, not as '
                 'a numeric score.'),
    }


def pool_equivalence_audit(pool_a: pd.DataFrame, pool_b: pd.DataFrame) -> Dict:
    """Full per-dimension matching report between two arm pools."""
    frames = {'a': pool_a, 'b': pool_b}
    seqs_a = pool_a['normalized_sequence'].tolist()
    seqs_b = pool_b['normalized_sequence'].tolist()
    labels_a = pool_a['activity_label'].astype(float)
    labels_b = pool_b['activity_label'].astype(float)
    return {
        'sample_size': sample_size_report(len(pool_a), len(pool_b)),
        'activity': activity_matching_report(labels_a, labels_b),
        'sequence': sequence_matching_report(seqs_a, seqs_b),
        'domain': domain_matching_report(
            frames, pool_a['source_dataset'].tolist(),
            pool_b['source_dataset'].tolist()),
    }


# ---------------------------------------------------------------------------
# feasibility audit (Section 4)
# ---------------------------------------------------------------------------


def activity_vs_composition_entanglement(labels: Sequence[float],
                                         sequences: Sequence[str],
                                         edges: Sequence[float]
                                         = ACTIVITY_EDGES) -> Dict:
    """
    Whether selecting rows by activity label also selects composition.
    GC is computed within each canonical activity bin of a single source.
    If mean GC varies substantially and monotonically across bins, then any
    activity-based arm construction is entangled with composition.
    """
    y = np.asarray(labels, dtype=float).ravel()
    gc = gc_content_array(list(sequences))
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (y >= lo) & (y <= hi)
        else:
            m = (y >= lo) & (y < hi)
        n = int(m.sum())
        rows.append({
            'bin': f'[{lo:.2f}, {hi:.2f}]',
            'n': n,
            'gc_mean': float(gc[m].mean()) if n else None,
            'gc_sd': float(gc[m].std(ddof=1)) if n > 1 else None,
        })
    valid = [r['gc_mean'] for r in rows if r['gc_mean'] is not None]
    return {
        'rows': rows,
        'gc_range_across_bins': (float(max(valid) - min(valid))
                                 if len(valid) > 1 else None),
        'monotonic_in_activity': bool(
            all(rows[i]['gc_mean'] is None or rows[i + 1]['gc_mean'] is None or
                rows[i + 1]['gc_mean'] >= rows[i]['gc_mean']
                for i in range(len(rows) - 1))),
        'entangled_warning': (
            'If GC correlates with activity within a source, activity-selected '
            'subsets differ in composition too; the dimensions remain '
            'confounded and must be reported as such.'),
    }


def _same_domain_extra_data(available_frames: Dict[str, Dict]) -> Dict:
    """
    Structural check for ARM B: additional training rows whose SOURCE and
    EXPERIMENTAL CONTEXT equal the canonical domain. Arm B requires extra
    examples with the canonical sequence/domain distribution, so only rows
    from a domain-identical source qualify.
    """
    extra = available_frames['additional']
    canonical = available_frames['canonical']
    qualifying = {}
    for src, info in extra.items():
        if info['source'] in {canonical['source']}:
            qualifying[src] = info['n']
    return {
        'same_domain_extra_rows_exist': bool(qualifying),
        'qualifying_sources': qualifying,
        'candidate_sources_checked': sorted(extra.keys()),
    }


def feasibility_audit(frames: Dict[str, Dict],
                      canonical_frame: pd.DataFrame,
                      additional_frames: Dict[str, pd.DataFrame],
                      augment_with: Optional[Dict] = None) -> Dict:
    """
    Section-4 audit. `frames` carries per-source meta:
      {'canonical': {'source': str, 'n': int, 'context': str},
       'additional': {src: {'source': str, 'n': int, 'context': str, ...}}}

    Returns a decision ('FEASIBLE' | 'INFEASIBLE'), the triggered STOP
    conditions, and all quantitative probes. Deterministic: same inputs,
    same decision.
    """
    aug = augment_with or {}
    same_domain = _same_domain_extra_data(frames)

    # Quantitative domain-distribution gap between canonical and each extra
    gap_probes = []
    can_seq = canonical_frame['normalized_sequence'].tolist()
    can_lab = canonical_frame['activity_label'].astype(float).values
    for src, df in additional_frames.items():
        gap_probes.append({
            'source': src,
            'sequence_gap': sequence_diversity_report(
                can_seq, df['normalized_sequence'].tolist(), add_name=src),
            'activity_gap': activity_matching_report(
                can_lab, df['activity_label'].astype(float).values),
        })

    # Within-extra activity<->composition entanglement
    entangle = {}
    for src, df in additional_frames.items():
        entangle[src] = activity_vs_composition_entanglement(
            df['activity_label'].astype(float).values,
            df['normalized_sequence'].tolist())

    # ---- decision
    triggered: List[int] = []
    notes = []

    # ARM B structural requirement: same-domain extra rows with larger n.
    if not same_domain['same_domain_extra_rows_exist']:
        triggered.append(1)   # no valid matched control constructible
        triggered.append(9)   # sample size cannot be controlled
        notes.append(
            'The only additional training source is DeepHF; no same-domain '
            'extra rows exist, so ARM B (sample size at constant canonical '
            'activity AND constant canonical sequence/domain) is '
            'structurally impossible. Sample count can therefore never be '
            'varied while holding the other two levers fixed.')

    # Quantitatively confirm no extra source can approximate canonical
    # composition even loosely (GC Cohen's d, k-mer JSD).
    for g in gap_probes:
        gc_d = g['sequence_gap']['gc']['cohens_d']
        jsd3 = g['sequence_gap']['kmer_3']['js_divergence']
        if abs(gc_d) > PROPOSED_TOLERANCES['gc_cohens_d'] or \
                jsd3 > PROPOSED_TOLERANCES['kmer3_js_divergence']:
            triggered.append(8)
            notes.append(
                f"{g['source']} differs from canonical composition "
                f"(GC d={gc_d:+.2f}, kmer3 JSD={jsd3:.3f}); any pool "
                'containing it changes both sequence diversity and activity '
                'coverage simultaneously.')
            break

    for src, e in entangle.items():
        if e['monotonic_in_activity'] and (e['gc_range_across_bins'] or 0) >= 0.04:
            triggered.append(7)
            triggered.append(8)
            notes.append(
                f"Within {src}, GC correlates monotonically with activity "
                f"(range {e['gc_range_across_bins']:.3f}); activity-selected "
                'subsets co-select composition. Activity coverage and '
                'sequence diversity cannot be separated, even inside one '
                'source.')
            break

    triggered = sorted(set(triggered))
    feasible = not triggered
    decision = 'FEASIBLE' if feasible else 'INFEASIBLE'

    return {
        'decision': decision,
        'triggered_stop_conditions': triggered,
        'stop_condition_texts': {str(k): STOP_CONDITIONS[k]
                                 for k in triggered},
        'notes': notes,
        'same_domain_extra_data': same_domain,
        'gap_probes': gap_probes,
        'within_source_entanglement': entangle,
        'conclusion': (
            'Phase 12 controlled isolation is not identifiable under the '
            'available data.' if not feasible
            else 'The data support a controlled ablation; proceed to arms.'),
        'augment': aug,
    }


def decide_gate(feasibility: Dict, hypothesis_evidence: Optional[Dict] = None
                ) -> Dict:
    """
    Section-27 decision gate. If the feasibility audit fails, the gate is F
    (controlled ablation infeasible). Otherwise the gate is derived from the
    (unused-in-this-phase) hypothesis evidence A..E.
    """
    if feasibility.get('decision') == 'INFEASIBLE':
        return {
            'gate': 'F',
            'label': DECISION_GATES['F'],
            'reason': ('The controlled 5-arm design cannot be constructed '
                       'with the available data (stop conditions '
                       f"{feasibility.get('triggered_stop_conditions')}). "
                       'Consequence for mechanism identifiability: E'),
        }
    h = hypothesis_evidence or {}
    supported = [k for k in ['C', 'B', 'A', 'D'] if h.get(k)]
    if len(supported) == 1:
        return {'gate': supported[0], 'label': DECISION_GATES[supported[0]],
                'reason': 'single supported lever'}
    if len(supported) > 1:
        return {'gate': 'D', 'label': DECISION_GATES['D']}
    return {'gate': 'E', 'label': DECISION_GATES['E']}


# ---------------------------------------------------------------------------
# deterministic arm construction (only reachable if the feasibility audit
# passes; on the available data it does not, so these helpers are exercised
# by tests on synthetic structures, not by the real runner)
# ---------------------------------------------------------------------------


def stratified_subsample(df: pd.DataFrame, n_target: int, seed: int,
                         edges: Sequence[float] = ACTIVITY_EDGES
                         ) -> pd.DataFrame:
    """
    Deterministic, activity-stratified subsample to (at most) n_target rows.
    Allocation is proportional to the available per-bin mass; the random draw
    inside every bin uses `seed` (numpy default_rng) and is fully
    reproducible. Selection uses label strateggala only - never any model or
    external signal.
    """
    if n_target <= 0:
        return df.iloc[0:0].reset_index(drop=True)
    df = df.reset_index(drop=True)
    y = df['activity_label'].astype(float).values
    edges = sorted(float(e) for e in edges)
    parts = []
    rng = np.random.default_rng(seed)
    avail = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (y >= lo) & (y <= hi)
        else:
            m = (y >= lo) & (y < hi)
        avail[i] = int(m.sum())
    total_avail = int(avail.sum())
    take = min(n_target, total_avail)
    quota = avail / total_avail
    assigned = np.floor(quota * take).astype(int)
    remainder = int(take - assigned.sum())
    # distribute the remainder to the bins with largest fractional loss
    frac_loss = quota * take - assigned
    order = np.argsort(-frac_loss)
    for j in range(remainder):
        assigned[order[j % len(order)]] += 1
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (y >= lo) & (y <= hi)
        else:
            m = (y >= lo) & (y < hi)
        idx = np.where(m)[0]
        if assigned[i] <= 0 or len(idx) == 0:
            continue
        pick = rng.choice(idx, size=int(assigned[i]), replace=False)
        parts.append(df.iloc[pick])
    if not parts:
        return df.iloc[0:0].reset_index(drop=True)
    return pd.concat(parts, ignore_index=True).sort_index().reset_index(drop=True)


def construct_arm_pools(canonical_split: pd.DataFrame,
                        extra_frames: Dict[str, pd.DataFrame],
                        sizes: Dict[str, int],
                        seed: int = 42,
                        edges: Sequence[float] = ACTIVITY_EDGES) -> Dict:
    """
    Deterministic construction of the pre-registered arms A..E under the
    assumption the feasibility audit passed.

    - A: canonical split, unmodified.
    - B: canonical + same-domain extra rows (enforced: only rows whose source
      equals the canonical source qualifiy), to size B. Raises if no
      same-domain extra data exists (STOP-9).
    - D: canonical + activity-stratified subsample of the extras matched to
      the canonical total, to size D (activity comparable to canonical).
    - C: canonical + activity-stratified subsample of the extras sized like D
      but deliberately biased to the high-activity tail (broader coverage).
    - E: canonical + all extra rows (frozen Phase 10 reference).
    """
    canonical_src = str(canonical_split['source_dataset'].iloc[0])
    a = canonical_split.copy()

    same_domain = {k: v for k, v in extra_frames.items()
                   if str(v['source_dataset'].iloc[0]) == canonical_src}
    if not same_domain:
        raise ValueError(
            "STOP-9: no same-domain extra data; ARM B cannot be constructed.")

    e = pd.concat([a] + list(extra_frames.values()), ignore_index=True)

    b_extra = pd.concat(list(same_domain.values()), ignore_index=True)
    n_b_extra = max(0, int(sizes['B']) - len(a))
    b_extra = stratified_subsample(b_extra, n_b_extra, seed, edges)
    b = pd.concat([a, b_extra], ignore_index=True)

    extra_all = pd.concat(list(extra_frames.values()), ignore_index=True)
    n_d_extra = max(0, int(sizes['D']) - len(a))
    d = pd.concat([a, stratified_subsample(extra_all, n_d_extra, seed,
                                           edges)], ignore_index=True)

    # C: same size as D, but keep the extra rows by seeding the EXTRAS row
    # near the high tail: stratify only on the upper bins where possible.
    n_c_extra = max(0, int(sizes.get('C', sizes['D'])) - len(a))
    if n_c_extra > 0:
        high = extra_all[extra_all['activity_label'].astype(float) > 0.6]
        low = extra_all[extra_all['activity_label'].astype(float) <= 0.6]
        ratio = min(1.0, len(high) / n_c_extra)
        n_high = int(round(n_c_extra * ratio)) if ratio < 0.8 else n_c_extra
        n_high = min(n_high, len(high))
        n_low = max(0, n_c_extra - n_high)
        parts = [stratified_subsample(high, n_high, seed, edges),
                 stratified_subsample(low, n_low, seed, edges)]
        c = pd.concat([a] + [p for p in parts if len(p)], ignore_index=True)
    else:
        c = a.copy()

    return {
        'A': a.reset_index(drop=True),
        'B': b.reset_index(drop=True),
        'C': c.reset_index(drop=True),
        'D': d.reset_index(drop=True),
        'E': e.reset_index(drop=True),
    }